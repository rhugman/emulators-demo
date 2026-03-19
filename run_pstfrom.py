import sys
import os
import shutil
import platform
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyemu
import flopy

from workflow_dsi import get_bins


def prep_forecasts(pst, model_times=False):
    pred_csv = os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth', "pred_data.csv")
    assert os.path.exists(pred_csv)
    pred_data = pd.read_csv(pred_csv)
    pred_data.set_index('site', inplace=True)

    if type(model_times) == bool:
        model_times = [float(i) for i in pst.observation_data.time.unique()]

    ess_obs_data = {}
    for site in pred_data.index.unique().values:
        site_obs_data = pred_data.loc[site, :].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:, "site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20, center=True, min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times, method="nearest")
        ess_obs_data[site] = sm_site_obs_data
    obs_data = pd.DataFrame(ess_obs_data)

    obs = pst.observation_data
    obs_names = [o for o in pst.obs_names if o not in pst.nnz_obs_names]

    time_str = obs_data.index.map(lambda x: f"time:{x}").values
    missing = []
    for col in obs_data.columns:
        if col.lower() == 'part_time':
            obs_sufix = col.lower()
        else:
            obs_sufix = col.lower() + "_" + time_str
        if type(obs_sufix) == str:
            obs_sufix = [obs_sufix]

        for string, oval, time in zip(obs_sufix, obs_data.loc[:, col].values, obs_data.index.values):
            if not any(string in obsnme for obsnme in obs_names):
                missing.append(string)
            else:
                obsnme = [ks for ks in obs_names if string in ks]
                if type(obsnme) == str:
                    obsnme = [obsnme]
                obsnme = obsnme[0]
                if obsnme == 'part_time':
                    oval = pred_data.loc['part_time', 'value']
                obs.loc[obsnme, "obsval"] = oval
    return


def process_secondary_obs(ws='.'):
    """Compute temporal-difference observations from model output CSVs.
    
    Note: imports are inside the function so that PstFrom can carry them
    into the generated forward_run.py.
    """
    import os
    import pandas as pd

    def write_tdif_obs(orgf, newf, ws='.'):
        df = pd.read_csv(os.path.join(ws, orgf), index_col='time')
        df = df - df.iloc[0, :]
        df.to_csv(os.path.join(ws, newf))

    write_tdif_obs('heads.csv', 'heads.tdiff.csv', ws)
    write_tdif_obs('sfr.csv', 'sfr.tdiff.csv', ws)
    print('Secondary observation files processed.')


def extract_hds_arrays_and_list_dfs():
    """Extract head arrays and list-budget DataFrames from MF6 outputs."""
    import flopy
    hds = flopy.utils.HeadFile("freyberg6_freyberg.hds")
    for it, t in enumerate(hds.get_times()):
        d = hds.get_data(totim=t)
        for k, dlay in enumerate(d):
            np.savetxt("hdslay{0}_t{1}.txt".format(k + 1, it + 1), d[k, :, :], fmt="%15.6E")

    lst = flopy.utils.Mf6ListBudget("freyberg6.lst")
    inc, cum = lst.get_dataframes(diff=True, start_datetime=None)
    inc.columns = inc.columns.map(lambda x: x.lower().replace("_", "-"))
    cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
    inc.index.name = "totim"
    cum.index.name = "totim"
    inc.to_csv("inc.csv")
    cum.to_csv("cum.csv")


def test_extract_hds_arrays(d):
    cwd = os.getcwd()
    os.chdir(d)
    extract_hds_arrays_and_list_dfs()
    os.chdir(cwd)


def run_pstfrom(template_ws):
    # --- Copy original model files ---
    org_d = os.path.join('templates', 'make_gmdsi_freyberg', 'monthly_model_files_1lyr_newstress')
    tmp_d = os.path.join('freyberg_mf6')
    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    shutil.copytree(org_d, tmp_d)
    get_bins(tmp_d)

    # --- Load and run the model once ---
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d)
    gwf = sim.get_model()
    pyemu.os_utils.run("mf6", cwd=tmp_d)
    pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim', cwd=tmp_d)

    # --- Spatial reference ---
    sr = pyemu.helpers.SpatialReference.from_namfile(
        os.path.join(tmp_d, "freyberg6.nam"),
        delr=gwf.dis.delr.array, delc=gwf.dis.delc.array)

    # --- Instantiate PstFrom ---
    start_datetime = "1-1-2008"
    pf = pyemu.utils.PstFrom(original_d=tmp_d,
                              new_d=template_ws,
                              remove_existing=True,
                              longnames=True,
                              spatial_reference=sr,
                              zero_based=False,
                              start_datetime=start_datetime,
                              echo=False)

    # --- Observations ---
    df = pd.read_csv(os.path.join(template_ws, "heads.csv"), index_col=0)
    hds_df = pf.add_observations("heads.csv",
                                  insfile="heads.csv.ins",
                                  index_cols="time",
                                  use_cols=list(df.columns.values),
                                  prefix="hds")

    df = pd.read_csv(os.path.join(template_ws, "sfr.csv"), index_col=0)
    sfr_df = pf.add_observations("sfr.csv",
                                  insfile="sfr.csv.ins",
                                  index_cols="time",
                                  use_cols=list(df.columns.values),
                                  prefix="sfr")

    # --- Geostatistical structures ---
    v_grid = pyemu.geostats.ExpVario(contribution=1.0, a=2000, anisotropy=1.0, bearing=0.0)
    grid_gs = pyemu.geostats.GeoStruct(variograms=v_grid, transform='log')

    v_pp = pyemu.geostats.ExpVario(contribution=1.0, a=30000, anisotropy=1.0, bearing=0.0)
    pp_gs = pyemu.geostats.GeoStruct(variograms=v_pp, transform='log')

    v_time = pyemu.geostats.ExpVario(contribution=1.0, a=180, anisotropy=1.0, bearing=0.0)
    temporal_gs = pyemu.geostats.GeoStruct(variograms=v_time, transform='none')

    # --- Helper: add multi-scale multiplier parameters ---
    ib = gwf.dis.idomain.array[0]

    def add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100, add_coarse=True):
        if isinstance(f, str):
            base = f.split(".")[1].replace("_", "")
        else:
            base = f[0].split(".")[1]
        pf.add_parameters(f, zone_array=ib, par_type="grid", geostruct=grid_gs,
                          par_name_base=base + "gr", pargp=base + "gr",
                          lower_bound=lb, upper_bound=ub, ult_ubound=uub, ult_lbound=ulb)
        pf.add_parameters(f, zone_array=ib, par_type="pilotpoints", geostruct=pp_gs,
                          par_name_base=base + "pp", pargp=base + "pp",
                          lower_bound=lb, upper_bound=ub, ult_ubound=uub, ult_lbound=ulb,
                          pp_space=5)
        if add_coarse:
            pf.add_parameters(f, zone_array=ib, par_type="constant", geostruct=grid_gs,
                              par_name_base=base + "cn", pargp=base + "cn",
                              lower_bound=lb, upper_bound=ub, ult_ubound=uub, ult_lbound=ulb)

    # --- Array parameters: Kh (with step-by-step for layer 1) ---
    f = 'freyberg6.npf_k_layer1.txt'
    pf.add_parameters(f, zone_array=ib, par_type="grid", geostruct=grid_gs,
                      par_name_base=f.split('.')[1].replace("_", "") + "gr",
                      pargp=f.split('.')[1].replace("_", "") + "gr",
                      lower_bound=0.2, upper_bound=5.0, ult_ubound=100, ult_lbound=0.01)
    pf.add_parameters(f, zone_array=ib, par_type="pilotpoints", geostruct=pp_gs,
                      par_name_base=f.split('.')[1].replace("_", "") + "pp",
                      pargp=f.split('.')[1].replace("_", "") + "pp",
                      lower_bound=0.2, upper_bound=5.0, ult_ubound=100, ult_lbound=0.01,
                      pp_space=5)
    pf.add_parameters(f, zone_array=ib, par_type="constant", geostruct=grid_gs,
                      par_name_base=f.split('.')[1].replace("_", "") + "cn",
                      pargp=f.split('.')[1].replace("_", "") + "cn",
                      lower_bound=0.2, upper_bound=5.0, ult_ubound=100, ult_lbound=0.01)

    # Add HK array as observations for post-processing convenience
    pf.add_observations(f, prefix="hk", zone_array=ib)

    # --- Array parameters: Ss, Sy, porosity ---
    tag = "sto_ss"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files[1:]:
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=1e-7, uub=1e-3)

    tag = "sto_sy"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    f = files[0]
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)

    tag = "ne_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files:
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)

    # --- Temporal correlation datetimes ---
    dts = pd.to_datetime(start_datetime) + pd.to_timedelta(
        np.cumsum(sim.tdis.perioddata.array["perlen"]), unit='d')

    # --- Recharge parameters (spatial + temporal) ---
    tag = "rch_recharge"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s: f for s, f in zip(sp, files)}
    sp.sort()
    files = [d[s] for s in sp]
    # spatial multipliers (shared across stress periods)
    add_mult_pars(files, lb=0.2, ub=5.0, ulb=0, uub=1e-3, add_coarse=False)
    # temporal constant multipliers (correlated in time)
    for f in files:
        kper = int(f.split('.')[1].split('_')[-1]) - 1
        pf.add_parameters(filenames=f, zone_array=ib, par_type="constant",
                          par_name_base=f.split('.')[1] + "tcn",
                          pargp=f.split('.')[1] + "tcn",
                          lower_bound=0.5, upper_bound=1.5,
                          ult_ubound=1e-3, ult_lbound=0,
                          datetime=dts[kper], geostruct=temporal_gs)

    # --- GHB parameters (conductance + head) ---
    tag = "ghb_stress_period_data"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files:
        name = 'ghbcond'
        pf.add_parameters(f, par_type="grid", geostruct=grid_gs,
                          par_name_base=name + "gr", pargp=name + "gr",
                          index_cols=[0, 1, 2], use_cols=[4],
                          lower_bound=0.1, upper_bound=10.0)
        pf.add_parameters(f, par_type="constant", geostruct=grid_gs,
                          par_name_base=name + "cn", pargp=name + "cn",
                          index_cols=[0, 1, 2], use_cols=[4],
                          lower_bound=0.1, upper_bound=10.0,
                          ult_lbound=0.01, ult_ubound=100)
        name = 'ghbhead'
        pf.add_parameters(f, par_type="grid", geostruct=grid_gs,
                          par_name_base=name + "gr", pargp=name + "gr",
                          index_cols=[0, 1, 2], use_cols=[3],
                          lower_bound=-2.0, upper_bound=2.0,
                          par_style="a", transform="none",
                          ult_lbound=32.5, ult_ubound=42)
        pf.add_parameters(f, par_type="constant", geostruct=grid_gs,
                          par_name_base=name + "cn", pargp=name + "cn",
                          index_cols=[0, 1, 2], use_cols=[3],
                          lower_bound=-2.0, upper_bound=2.0,
                          par_style="a", transform="none",
                          ult_lbound=32.5, ult_ubound=42)

    # --- WEL parameters (temporal constant + grid) ---
    files = [f for f in os.listdir(template_ws) if "wel_stress_period_data" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s: f for s, f in zip(sp, files)}
    sp.sort()
    files = [d[s] for s in sp]
    for f in files:
        kper = int(f.split('.')[1].split('_')[-1]) - 1
        pf.add_parameters(filenames=f, index_cols=[0, 1, 2], use_cols=[3],
                          par_type="constant", par_name_base="welcst", pargp="welcst",
                          upper_bound=4, lower_bound=0.25,
                          datetime=dts[kper], geostruct=temporal_gs)
        pf.add_parameters(filenames=f, index_cols=[0, 1, 2], use_cols=[3],
                          par_type="grid", par_name_base="welgrd", pargp="welgrd",
                          upper_bound=4, lower_bound=0.25,
                          datetime=dts[kper])

    # --- SFR parameters (conductance + inflow) ---
    tag = "sfr_packagedata"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    assert len(files) == 1
    f = files[0]
    name = "sfrcond"
    pf.add_parameters(f, par_type="grid", geostruct=grid_gs,
                      par_name_base=name + "gr", pargp=name + "gr",
                      index_cols=[0, 2, 3], use_cols=[9],
                      lower_bound=0.1, upper_bound=10.0)
    pf.add_parameters(f, par_type="constant", geostruct=grid_gs,
                      par_name_base=name + "cn", pargp=name + "cn",
                      index_cols=[0, 2, 3], use_cols=[9],
                      lower_bound=0.1, upper_bound=10.0,
                      ult_lbound=0.001, ult_ubound=100)

    files = [f for f in os.listdir(template_ws) if "sfr_perioddata" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s: f for s, f in zip(sp, files)}
    sp.sort()
    files = [d[s] for s in sp]
    for f in files:
        kper = int(f.split('.')[1].split('_')[-1]) - 1
        pf.add_parameters(filenames=f, index_cols=[0], use_cols=[2],
                          par_type="grid", par_name_base="sfrgr", pargp="sfrgr",
                          upper_bound=10, lower_bound=0.1,
                          datetime=dts[kper], geostruct=temporal_gs)

    # --- Initial conditions ---
    files = [f for f in os.listdir(template_ws) if "ic_strt" in f and f.endswith(".txt")]
    for f in files:
        base = f.split(".")[1].replace("_", "")
        pf.add_parameters(f, par_type="grid", par_style="d",
                          pargp=base, par_name_base=base,
                          upper_bound=50, lower_bound=15,
                          zone_array=ib, transform="none")

    # --- Build PST and add system commands ---
    pst = pf.build_pst()
    pf.mod_sys_cmds.append("mf6")
    pf.mod_sys_cmds.append("mp7 freyberg_mp.mpsim")

    # --- Post-processing functions ---
    pf.add_py_function("run_pstfrom.py", "extract_hds_arrays_and_list_dfs()", is_pre_cmd=False)

    test_extract_hds_arrays(template_ws)
    files = [f for f in os.listdir(template_ws) if f.startswith("hdslay")]
    for f in files:
        pf.add_observations(f, prefix=f.split(".")[0], obsgp=f.split(".")[0])
    for f in ["inc.csv", "cum.csv"]:
        df = pd.read_csv(os.path.join(template_ws, f), index_col=0)
        pf.add_observations(f, index_cols=["totim"], use_cols=list(df.columns.values),
                            prefix=f.split('.')[0], obsgp=f.split(".")[0])

    process_secondary_obs(ws=template_ws)
    pf.add_py_function("run_pstfrom.py", "process_secondary_obs(ws='.')", is_pre_cmd=False)

    df = pd.read_csv(os.path.join(template_ws, "sfr.tdiff.csv"), index_col=0)
    pf.add_observations("sfr.tdiff.csv", insfile="sfr.tdiff.csv.ins",
                        index_cols="time", use_cols=list(df.columns.values), prefix="sfrtd")
    df = pd.read_csv(os.path.join(template_ws, "heads.tdiff.csv"), index_col=0)
    pf.add_observations("heads.tdiff.csv", insfile="heads.tdiff.csv.ins",
                        index_cols="time", use_cols=list(df.columns.values), prefix="hdstd")

    # --- Rebuild PST with all observations and commands ---
    pst = pf.build_pst()

    # --- Add MODPATH endpoint observations from custom INS file ---
    out_file = "freyberg_mp.mpend"
    ins_file = out_file + ".ins"
    with open(os.path.join(template_ws, ins_file), 'w') as f:
        f.write("pif ~\n")
        f.write("l7 w w w w !part_status! w w !part_time!\n")
    pst.add_observations(ins_file=os.path.join(template_ws, ins_file),
                         out_file=os.path.join(template_ws, out_file),
                         pst_path='.')
    obs = pst.observation_data
    obs.loc[obs.obsnme == 'part_status', 'obgnme'] = 'part'
    obs.loc[obs.obsnme == 'part_time', 'obgnme'] = 'part'

    # --- Fix additive parameters with zero initial value ---
    head_pargps = [i for i in pst.adj_par_groups if 'head' in i]
    pst.parameter_groups.loc[head_pargps, 'inctyp'] = 'absolute'
    par = pst.parameter_data
    par_names = par.loc[par.parval1 == 0].parnme
    offset = -10
    par.loc[par_names, 'offset'] = offset
    par.loc[par_names, ['parval1', 'parlbnd', 'parubnd']] -= offset

    # --- Forecasts ---
    forecasts = [
        'oname:sfr_otype:lst_usecol:tailwater_time:4383.5',
        'oname:sfr_otype:lst_usecol:headwater_time:4383.5',
        'oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5',
        'part_time'
    ]
    pst.pestpp_options['forecasts'] = forecasts
    pst.pestpp_options["overdue_giveup_fac"] = 3
    pst.pestpp_options["overdue_giveup_minutes"] = 0.5

    # --- Write, verify, and generate prior ensemble ---
    pst.write(os.path.join(template_ws, 'pest.pst'), version=2)
    pyemu.os_utils.run('pestpp-ies pest.pst', cwd=template_ws)

    if pf.pst.npar < 35000:
        cov = pf.build_prior(fmt='coo', filename=os.path.join(template_ws, "prior_cov.jcb"))

    pe = pf.draw(num_reals=1000, use_specsim=True)
    pe.enforce()
    pe.to_binary(os.path.join(template_ws, "prior_pe.jcb"))
    assert pe.shape[1] == pst.npar

    pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'
    pst.write(os.path.join(template_ws, 'pest.pst'), version=2)

    return


def set_weights(t_d):
    pst_file = "pest.pst"
    pst = pyemu.Pst(os.path.join(t_d, pst_file))
    obs = pst.observation_data

    # Zero all weights, then selectively assign
    obs.loc[:, 'weight'] = 0

    # --- Load and smooth-sample measured observation data ---
    obs_csv = os.path.join('templates', 'make_gmdsi_freyberg', "obs_data.csv")
    assert os.path.exists(obs_csv)
    obs_data = pd.read_csv(obs_csv)
    obs_data.set_index('site', inplace=True)

    model_times = pst.observation_data.time.dropna().astype(float).unique()
    obs_sites = obs_data.index.unique().tolist()

    ess_obs_data = {}
    for site in obs_sites:
        site_obs_data = obs_data.loc[site, :].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:, "site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20, center=True, min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times, method="nearest")
        ess_obs_data[site] = sm_site_obs_data
    ess_obs_data = pd.DataFrame(ess_obs_data)

    # --- Helper: update obsvals and set weights for historical period ---
    def update_pst_obsvals(obs_names, obs_data_df):
        org_nnzobs = pst.nnz_obs
        time_str = obs_data_df.index.map(lambda x: f"time:{x}").values
        missing = []
        for col in obs_data_df.columns:
            obs_sufix = col.lower() + "_" + time_str
            for string, oval, time in zip(obs_sufix, obs_data_df.loc[:, col].values, obs_data_df.index.values):
                if not any(string in obsnme for obsnme in obs_names):
                    if string.startswith("trgw-2"):
                        pass
                    else:
                        missing.append(string)
                else:
                    obsnme = [ks for ks in obs_names if string in ks]
                    assert len(obsnme) == 1, string
                    obsnme = obsnme[0]
                    obs.loc[obsnme, "obsval"] = oval
                    if time > 3652.5 and time <= 4018.5:
                        obs.loc[obsnme, "weight"] = 1.0
        if len(missing) == 0:
            print('All good.')
            print('Number of new nonzero obs:', pst.nnz_obs - org_nnzobs)
            print('Number of nonzero obs:', pst.nnz_obs)
        else:
            raise ValueError('The following obs are missing:\n', missing)

    # --- Update absolute observations ---
    obs_names = obs.loc[obs.oname.isin(['hds', 'sfr']), 'obsnme']
    update_pst_obsvals(obs_names, ess_obs_data)

    # --- Update model output CSVs with measured values for secondary obs ---
    def update_obs_csv(obs_csv_path):
        obsdf = pd.read_csv(obs_csv_path, index_col=0)
        check = obsdf.copy()
        for col in ess_obs_data.columns:
            if col in obsdf.columns:
                obsdf.loc[:, col] = ess_obs_data.loc[:, col]
        obsdf.to_csv(obs_csv_path)
        obsdf = pd.read_csv(obs_csv_path, index_col=0)
        assert (obsdf.index == check.index).all()
        return obsdf

    update_obs_csv(os.path.join(t_d, "sfr.csv"))
    update_obs_csv(os.path.join(t_d, "heads.csv"))
    process_secondary_obs(ws=t_d)

    # --- Update secondary (tdiff) observations ---
    diff_obsdict = {
        'sfrtd': "sfr.tdiff.csv",
        'hdstd': "heads.tdiff.csv",
    }
    for keys, value in diff_obsdict.items():
        obs_names = obs.loc[obs.oname.isin([keys]), 'obsnme']
        obs_csv_df = pd.read_csv(os.path.join(t_d, value), index_col=0)
        usecols = list(set(map(str.upper, obs.loc[pst.nnz_obs_names, 'usecol'].unique())) & set(obs_csv_df.columns.tolist()))
        obs_csv_df = obs_csv_df.loc[:, usecols]
        org_nnz_obs_names = pst.nnz_obs_names
        update_pst_obsvals(obs_names, obs_csv_df)
        assert (pst.nnz_obs - len(org_nnz_obs_names)) == 12 * len(usecols), \
            [i for i in pst.nnz_obs_names if i not in org_nnz_obs_names]

    # --- Balance observation weights across groups ---
    pst.write(os.path.join(t_d, pst_file), version=2)
    pyemu.os_utils.run("pestpp-ies.exe {0}".format(pst_file), cwd=t_d)
    pst = pyemu.Pst(os.path.join(t_d, pst_file))
    obs = pst.observation_data

    contrib_per_group = pst.phi / float(len(pst.nnz_obs_groups))
    balanced_groups = {grp: contrib_per_group for grp in pst.nnz_obs_groups}
    pst.adjust_weights(obsgrp_dict=balanced_groups)

    pst.observation_data.loc[pst.nnz_obs_names, 'observed'] = 1

    # --- Assign measurement noise standard deviations ---
    obs.loc[:, "standard_deviation"] = np.nan
    hds_obs = [o for o in pst.nnz_obs_names if "oname:hds_" in o]
    assert len(hds_obs) > 0
    obs.loc[hds_obs, "standard_deviation"] = 0.3

    hdstd_obs = [o for o in pst.nnz_obs_names if "oname:hdstd_" in o]
    assert len(hdstd_obs) > 0
    obs.loc[hdstd_obs, "standard_deviation"] = 0.001

    sfr_obs = [o for o in pst.nnz_obs_names if "oname:sfr_" in o]
    assert len(sfr_obs) > 0
    obs.loc[sfr_obs, "standard_deviation"] = obs.loc[sfr_obs, "obsval"] * 0.15

    sfrtd_obs = [o for o in pst.nnz_obs_names if "oname:sfrtd_" in o]
    assert len(sfrtd_obs) > 0
    obs.loc[sfrtd_obs, "standard_deviation"] = obs.loc[sfrtd_obs, "obsval"] * 0.15

    pst.write(os.path.join(t_d, pst_file), version=2)


if __name__ == "__main__":
    run_pstfrom()
    set_weights()
