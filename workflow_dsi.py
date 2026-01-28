import pyemu
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import flopy
from pyemu.emulators.dsi import DSI
import platform

def get_bins(dst,bin_path="bin",fnames=['pestpp-ies']):
    if platform.system() == "Windows":
        bin_path = os.path.join(bin_path, "win")
    elif platform.system() == "Linux":
        bin_path = os.path.join(bin_path, "linux")
    elif platform.system() == "Darwin":
        bin_path = os.path.join(bin_path, "mac")

    for f in fnames:
        shutil.copy2(os.path.join(bin_path,f),os.path.join(dst,f)) 

    return


def run_training_ensemble(t_d="pst_template",m_d='master_prior',num_workers=15,ies_num_reals=1000,noptmax=-1):

    org_t_d = os.path.join("templates","freyberg_template")
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_t_d,t_d)



    pst = pyemu.Pst(os.path.join(t_d,"freyberg_mf6.pst"))

    # set the number of realizations; relies on existence of the prior jcb
    pst.pestpp_options["ies_num_reals"] = ies_num_reals

    # verify that the parameter ensemble file exists
    filename = os.path.join(t_d, "prior_pe.jcb")
    assert os.path.exists(filename), f"parameter ensemble file {filename} not found"
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=filename)
    assert pe.shape[0] >= ies_num_reals, f"ies_num_reals={ies_num_reals} but only {pe.shape[0]} realizations in the ensemble"

    # reduce the noise in pumping rates
    par = pst.parameter_data
    # make cst wel pars 1.0
    pe.loc[:, par.loc[par.pname=="welcst"].parnme] = 1

    # make sure the pumping rates are the same for each well in the future; and cosntant in the past
    welgrd = par.loc[par.pargp=="welgrd"].copy()
    welgrd.inst = welgrd.inst.astype(int)
    welgrd["wellid"] = welgrd.apply(lambda x: f"{x.idx1}_{x.idx2}",axis=1)
    welgrd.sort_values(by=["wellid","inst"],inplace=True)
    t_hist = welgrd.inst.unique()[12]
    for i, (wellid, df) in enumerate(welgrd.groupby("wellid")):
        parnmes = df.parnme.values
        parnme_hist = df.loc[df.inst<=t_hist].parnme
        pe.loc[:,parnme_hist] = 1.0 #+ 0.1 * np.random.randn(pe.shape[0],len(parnme_hist))

        parnme_future = df.loc[df.inst>t_hist].parnme
        pe.loc[:,parnme_future] = 1.0 # set same future as past; 

        #NOTE: use the code below to generate training data for optimisation...
        #val = np.random.uniform(0.25,4.,pe.shape[0])
        #for j, parnme in enumerate(parnme_future):
        #    pe.loc[:,parnme] = val #pe._df[parnme_future[0]].values

    pe.to_binary(os.path.join(t_d, pst.pestpp_options['ies_parameter_ensemble']))

    # make sure obs weights = 1/stdv
    obs = pst.observation_data
    obs.loc[obs.weight>0,"weight"] = 1.0 / obs.loc[obs.weight>0,"standard_deviation"]
    assert obs.weight.sum()>0, "no non-zero obs weights found"
    #drop time-diff observations
    obs.loc[obs.oname.isin(['hdstd','sfrtd']), 'weight'] = 0.0

    # set sfr obs weights to a fixed value, because we are going to be changing the obsval 
    obs.loc[(obs.usecol=='gage-1') & (obs.weight>0),'weight'] = 1 / 100 #arbitrary

    # add some more obs data
    times = obs.loc[obs.weight>0].time.unique()
    obs.loc[(obs.time.isin(times)) & (obs.oname=='hds'), 'weight'] = 2.0
    #obs.loc[(obs.time.isin(times)) & (obs.oname=='sfr'), 'weight'] = 1/10


    # reduce the number of lost reals
    pst.pestpp_options["overdue_giveup_fac"] = 1e30
    pst.pestpp_options["overdue_giveup_minutes"] = 1e30
    pst.pestpp_options["save_binary"] = True    

    pst.control_data.noptmax = noptmax

    f = "pestpp-ies"
    #shutil.copy2(os.path.join("bin","mac",f),os.path.join(t_d,f))
    get_bins(t_d,bin_path="bin",fnames=['pestpp-ies'])

    # rewirte and deploy
    pst.write(os.path.join(t_d,"pest.pst"),version=2)
    pyemu.os_utils.start_workers(t_d,"pestpp-ies","pest.pst",
                                 num_workers=num_workers,worker_root=".",
                                    master_dir=m_d)

    return


def plot_well_tseries(pst):
    pst.try_parse_name_metadata
    pe = pst.ies.parens0.copy()
    par = pst.parameter_data
    obs = pst.observation_data.copy()
    obs.time = obs.time.astype(float)
    wobs = par.loc[par.pname=="welgrd"].copy()
    wobs.inst = wobs.inst.astype(int)
    wobs["wellid"] = wobs.apply(lambda x: f"{x.idx1}_{x.idx2}",axis=1)
    wobs.sort_values(by=["wellid","inst"],inplace=True)
    wobs.sort_values(by='inst',inplace=True)
    #fwells = wobs.loc[wobs.inst>12].obsnme.values
    wobs

    fig,ax = plt.subplots(1,1,figsize=(6,3))
    for w in wobs.wellid.unique():
        inst = wobs.loc[wobs.wellid==w,"inst"].values
        cols = wobs.loc[wobs.wellid==w,"parnme"].tolist()  
        times = obs.time.astype(float).unique()
        [ax.plot(inst, pe.loc[i,cols],color='0.5',lw=1,alpha=0.3) for i in pe.index];
    return

def load_training_data(md="master_prior"):
    pst = pyemu.Pst(os.path.join(md,"pest.pst"))

    obs = pst.observation_data

    oe = pst.ies.obsen0.copy()

    # drop physically unrealistic realizations
    sim = flopy.mf6.MFSimulation.load(sim_ws=md,load_only=['dis'],verbosity_level=0)
    top = sim.get_model().dis.top.array
    botm = sim.get_model().dis.botm.array

    obs = pst.observation_data
    obgnmes = [o for o in obs.obgnme.unique() if "hdslay1_" in o]
    obsnmes = obs.loc[obs.obgnme.isin(obgnmes)].obsnme.values
    # find any row that contians values greater than top
    oe_ = oe.copy()
    oe_.replace(1e30, np.nan, inplace=True)
    idx_keep = oe_.loc[:,obsnmes].apply(lambda x: x.max() < 5+top.max(), axis=1)
    oe_ = oe_.loc[idx_keep,:]
    print(idx_keep.sum())

    idx_keep = oe_.loc[:,obsnmes].apply(lambda x: x.min() > botm.min(), axis=1)
    print(idx_keep.sum())
    oe_ = oe_.loc[idx_keep,:]
    print(oe_.shape)

    return oe_, pst

def choose_inconvenient_truth(pst, oe, usecol="gage-1",ascending=True):

    obs = pst.observation_data
    obs.time = obs.time.astype(float)

    forecasts = obs.loc[(obs.usecol==usecol) & 
                        (obs.oname.isin(['sfr','hds'])) &
                        (obs.time<=obs.time.unique()[12])
                        ].obsnme

    check = oe.loc[:,forecasts].copy()
    sorted_index = check.sum(axis=1).sort_values(ascending=ascending).index.values
    truth_real = sorted_index[-1]
    sorted_index = sorted_index[:-1]

    return truth_real, sorted_index

def plot_oe_tseries(pst,oe_dict, truth_data=None, obgnmes=None):
    figures = []
    obs = pst.observation_data

    # Plot ensemble forecast with uncertainty bounds
    nzobs = obs.loc[obs.weight > 0]
    if obgnmes is None:
        obgnmes = obs.obgnme.unique()
    for group_name in obgnmes:
        #if group_name in groups:
        try:
            forecast_obs = obs.loc[obs.obgnme==group_name].obsnme.tolist()#groups[group_name]
            forecast_times = obs.loc[forecast_obs].time.astype(float)
            historical_time_threshold = forecast_times.values[12]
        except:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot truth
            
        if truth_data is None:
            truth_values = obs.loc[forecast_obs].obsval.values.flatten()
        else:
            truth_values = truth_data.loc[forecast_obs].values.flatten()
        ax.plot(forecast_times, truth_values, 'r-', linewidth=2, 
                label='Truth', markersize=4, zorder=10)
        

        nzobsnmes = obs.loc[(obs.obgnme==group_name) & (obs.weight > 0)].obsnme.tolist()#groups[group_name]
        nztimes = obs.loc[nzobsnmes].time.astype(float)
        ax.plot(nztimes, obs.loc[nzobsnmes].obsval.values, 'k-', linewidth=2, 
                label='measured', marker='o', markersize=4, zorder=10)

        zorder=1
        for c,oe in oe_dict.items():
            [ax.plot(forecast_times,oe.loc[i,forecast_obs],c=c,zorder=zorder,alpha=0.3,lw=1) for i in oe.index]
            zorder += 1
        #[ax.plot(forecast_times,proe.loc[i,forecast_obs],c='0.5',zorder=0,alpha=0.3,lw=1) for i in ptoe.index]


        #[ax.plot(forecast_times,train_data.loc[i,forecast_obs],c='0.5',zorder=0,alpha=0.3,lw=1) for i in train_data.index]



        # Add vertical line at historical/forecast boundary
        ax.axvline(x=historical_time_threshold, color='gray', linestyle='--', 
                        alpha=0.7, label='Historical/Forecast Boundary')
        
        ax.set_xlabel('Time (days)')
        flow_names = ['tailwater','headwater','gage-1']
        if any(fn in group_name for fn in flow_names):
            ax.set_ylabel('Rate (m³/d)')
        else:
            ax.set_ylabel('Head (m)')

        if "gage-1" in group_name:
            ax.set_ylim(0,10000)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(group_name)
        
        fig.tight_layout()
        figures.append(fig)

                
        #else:
        #    print("No suitable forecast group found for plotting")


    return figures

def get_fitbygroups(pst):

    obs_train = pst.observation_data.copy()

    # Define time threshold for historical vs forecast data
    obs_train['time'] = obs_train['time'].astype(float)
    historical_time_threshold = obs_train.time.unique()[:12][-1]  # Use first 12 time points for fitting

    # Define fit_groups: observations used to compute min/max for scaling (historical data only)
    fit_groups = {
        o: obs_train.loc[(obs_train.obgnme == o) & 
                        (obs_train.time <= historical_time_threshold) 
                        & (obs_train.weight > 0)
                        ].obsnme.values 
        for o in obs_train.obgnme.unique()
    }
    # drop keys if empty
    fit_groups = {k: v for k, v in fit_groups.items() if v.size > 0}

    # Define groups: all observations in each group (historical + forecast)
    groups = {
        o: obs_train.loc[(obs_train.obgnme == o)].obsnme.values 
        for o in list(fit_groups.keys())
    }

    print("Groups defined for row-wise scaling:")
    for group_name, group_obs in groups.items():
        fit_obs = fit_groups[group_name]
        print(f"  {group_name}: {len(group_obs)} total obs, {len(fit_obs)} for fitting")
    return groups, fit_groups

def drop_obs(pst,m_d):
    # drop obs to ncols
    drop_list = [f for f in pst.instruction_files if f.startswith("hdslay")]
    drop_list.extend([f for f in pst.instruction_files if ".npf_k_layer1." in f])
    drop_list.append("inc.csv.ins")
    drop_list.append("cum.csv.ins")
    drop_list.extend([f for f in pst.instruction_files if ".tdiff." in f])
    drop_list.extend(['./freyberg_mp.mpend.ins'])
    for o in drop_list:
        pst.drop_observations(os.path.join(m_d,o),pst_path='.')
    return

def add_wellpars_to_data(pst,data,pe=None):
    # Load parameter ensemble
    if pe is None:
        pe = pst.ies.parens0.copy()

    obsdata = pst.observation_data.copy()
    obsdata.obsval = obsdata.obsval.astype(float)
    obsdata.weight = obsdata.weight.astype(float)

    print(f"Parameter ensemble shape: {pe.shape}")
    # Process well parameters (if available)
    par = pst.parameter_data
    welcst_params = par.loc[par.pargp == "welcst"] if "welcst" in par.pargp.values else pd.DataFrame()
    welgrd_params = par.loc[par.pargp == "welgrd"] if "welgrd" in par.pargp.values else pd.DataFrame()

    if not welcst_params.empty and not welgrd_params.empty:
        # Process well parameters as in original notebook
        welcst_params = welcst_params.copy()
        welgrd_params = welgrd_params.copy()
        welcst_params['inst'] = welcst_params['inst'].astype(int)
        welgrd_params['inst'] = welgrd_params['inst'].astype(int)
        welgrd_params["wellid"] = welgrd_params.apply(lambda x: f"{x.idx1}_{x.idx2}",axis=1)

        
        cols = welgrd_params.parnme.tolist() #+ welcst_params.parnme.tolist()
        pe_well = pe.loc[:, cols].copy()
        
        # Apply constant multipliers to grid parameters
        for inst in welcst_params.inst.unique():
            welnames = welgrd_params.loc[welgrd_params.inst == inst, "parnme"].tolist()
            cstnames = welcst_params.loc[welcst_params.inst == inst, "parnme"].tolist()
            if welnames and cstnames:
                pe_well.loc[:, welnames] = pe.loc[:, welnames].values * pe.loc[:, cstnames].values
        
        # Add well parameters to the training data
        for col in pe_well.columns:
            _ = pd.DataFrame(index=[col],columns=obsdata.columns)
            _.loc[col, "obsnme"] = col
            _.loc[col, "obgnme"] = 'wel'
            _.loc[col, "wellid"] = welgrd_params.loc[col,'wellid']
            _.loc[col, "weight"] = 1/0.01
            _.loc[col, "standard_deviation"] = 0.0
            _.loc[col, "obsval"] = float(welgrd_params.loc[col,'parval1'])
            _.dropna(axis=1,how='all',inplace=True)
            obsdata = pd.concat([obsdata, _], ignore_index=False)
            #if col not in data.columns:
            #    vals = pe_well.loc[data.index.values, col].astype(float).values
            #    assert len(vals) == data.shape[0], f"mismatch adding well param {col}"
            #    assert not np.any(np.isnan(vals)), f"NaNs found adding well param {col}"
            #    data.loc[:,col] = vals
        data.loc[:,pe_well.columns] = pe_well.loc[data.index,:].values

        well_input_cols = pe_well.columns.tolist()
        print(f"Added {len(well_input_cols)} well parameters as inputs")
    else:
        well_input_cols = []
        print("No well parameters found")


    return data, obsdata, well_input_cols

def prepare_dsi_dir(obsdata,train_data,groups,fit_groups,transforms=None):
    obsdata.obsval = obsdata.obsval.astype(float)
    train_data = train_data.astype(float)

    if groups is None or fit_groups is None:
        dsi = DSI(
                data=train_data.copy(), 
                transforms=transforms, 
                energy_threshold=1.,
                 verbose=True)
    else:
        dsi = DSI(pst=obsdata, 
                data=train_data.copy(), 
                transforms=transforms, energy_threshold=1.,
                rowwise_groups=groups, 
                rowwise_fit_groups=fit_groups, 
                feature_range=(-1, 1), verbose=True)        
    dsi.fit()
    if groups is not None:
        dsi._prefit_truth_rowwise_scaler()
        assert dsi._truth_rowwise_scaler is not None, "row-wise scaler not set!"
    t_d="dsi_template"
    pst = dsi.prepare_pestpp(t_d=t_d,observation_data=obsdata)

    #shutil.copy2("pestpp-ies", os.path.join(t_d,"pestpp-ies"))
    get_bins(t_d,bin_path="bin",fnames=['pestpp-ies'])
    # Predict using zero latent vector to get the mean shape in scaled space
    pvals = np.zeros_like(dsi.s)
    pred = dsi.predict(pvals)

    pst.control_data.noptmax = 3
    pst.pestpp_options["ies_num_reals"] = 200
    #pst.pestpp_options["ies_multimodal_alpha"] = 0.2
    pst.pestpp_options['ies_no_noise'] = False
    
    pst.pestpp_options["ies_drop_conflicts"] = False

    noise = pd.DataFrame(columns=dsi.data.columns,
                         index=np.arange(pst.pestpp_options["ies_num_reals"]))
    obs = pst.observation_data
    # for each timeseries, add a random offset
    for grp in obs.obgnme.unique():
        obsnmes = obs.loc[(obs.obgnme==grp) & (obs.weight>0)].obsnme.tolist()
        stdv = obs.loc[obsnmes].weight / 10 
        #if "gage" in grp:
        #    stdv = 100
        offset = np.random.normal(0, stdv, size=(pst.pestpp_options["ies_num_reals"], len(obsnmes)))
        noise.loc[:, obsnmes] = obs.loc[obsnmes].obsval.values + offset
    noise.to_csv(os.path.join(t_d, "noise.csv"))
    pst.pestpp_options["ies_obs_en"] = "noise.csv"

    pst.write(os.path.join(t_d, "dsi.pst"),version=2)


    return dsi, pst

def run_dsi(t_d="dsi_template",tag="row"):
    if tag=="row":
        dsi = DSI.load(os.path.join(t_d,"dsi.pickle"))
    elif tag.startswith("standard"):
        dsi = DSI.load(os.path.join(t_d,"dsi.pickle"))
    pvals = pd.read_csv(os.path.join(t_d, "dsi_pars.csv"), index_col=0)
    md = f"master_dsi"+f"_{tag}"
    num_workers = 15
    worker_root = "."
    pyemu.os_utils.start_workers(
        t_d,"pestpp-ies","dsi.pst", num_workers=num_workers,
        worker_root=worker_root, master_dir=md, #port=_get_port(),
        ppw_function=pyemu.helpers.dsi_pyworker,
        ppw_kwargs={
            "dsi": dsi, "pvals": pvals,
        }
    )
    cleanup_pypestworker_logs(".")
    return

def plot_results():

    oe_dict= {}#"0.5":data}

    pst = pyemu.Pst(os.path.join("master_dsi_standard","dsi.pst"))

    #oe_dict["0.5"] = pst.ies.obsen0.copy()
    oe_dict["b"] = pst.ies.obsen3.copy()

    pst = pyemu.Pst(os.path.join("master_dsi_row","dsi.pst"))
    #oe_dict["0.5"] = pst.ies.obsen0.copy()
    oe_dict["c"] = pst.ies.obsen3.copy()


    figures = plot_oe_tseries(pst,oe_dict,)

    # save as pdf
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages('results.pdf') as pdf:
        # plot each figure
        for f in figures:
            pdf.savefig(f)
        pdf.close()

    return

def update_obsdata_with_truth(obsdata,truth_real,oe):
    cols = list(set(obsdata.obsnme) & set(oe.columns))
    obsdata.loc[cols,"obsval"] = oe.loc[truth_real,cols].values
    return obsdata

def iah_plots():

    pst = pyemu.Pst(os.path.join("master_dsi_standard_easy","dsi.pst"))
    pst.try_parse_name_metadata()
    obs = pst.observation_data
    obs.time = obs.time.astype(float)
    obs.sort_values(by=["usecol","time"],inplace=True)
    #obs.loc[oe.columns, "obsval"] = oe.loc[truth_real,:].values
    forecasts = obs.loc[(obs.usecol=="tailwater") & (obs.oname=='sfr')].obsnme
    times = obs.loc[forecasts].time.astype(float)


    fig,axs = plt.subplots(2,1,figsize=(6,6),sharex=True)

    obsnmes = obs.loc[(obs.usecol=="trgw-0-13-10") & (obs.oname=='hds') & (obs.weight>0)].obsnme
    t = obs.loc[obsnmes].time.astype(float)
    ax = axs[0]
    #ax.plot(t, oe._df.loc[truth_real,obsnmes].values,c="k",linestyle="",marker='o')
    ax.plot(t, obs.loc[obsnmes].obsval.values,c="k",linestyle="",marker='o')
    ax.set_title("Monitoring bore")
    ax.set_ylabel("Head (m)")
    ax.set_ylim(33,36)

    obsnmes = obs.loc[(obs.usecol=="gage-1") & (obs.oname=='sfr') & (obs.weight>0)].obsnme
    t = obs.loc[obsnmes].time.astype(float)
    ax = axs[1]
    ax.plot(t, obs.loc[obsnmes].obsval.values,c="k",linestyle="",marker='o')
    ax.set_title("Stream gage")
    ax.set_ylabel("Flow (m3/d)")
    ax.set_ylim(0)

   #ax = axs[-1]
   #ax.plot(times, obs.loc[forecasts].obsval.values,c="r")
   #ax.set_title("SW-GW Exchange")
   #ax.set_ylabel("Flux (m3/d)")
   ##xmin,xmax = ax.get_xlim()
   ##ax.hlines(0,xmin,xmax,linestyles="-",color="0.5")
   ##ax.set_xlim(xmin,xmax)


    for ax in axs.flatten():
        ymin,ymax = ax.get_ylim()
        ax.vlines([max(t)],ymin,ymax,linestyles="--",color="0.5")
        
        #ax.text(max(t)-10,ymax*0.9,"History",rotation=0,ha="right",va="top",color="0.5")
        #ax.text(max(t)+60,ymax*0.9,"Future",rotation=0,ha="right",va="top",color="0.5")
        ax.set_ylim(ymin,ymax)
        ax.set_xlim([min(times), max(times)])
    ax.set_xlabel("Time (d)")
    fig.tight_layout()
    fig.savefig("iah_truth.png",dpi=300)


def cleanup_pypestworker_logs(t_d):
    log_files = [f for f in os.listdir(t_d) if f.startswith("pypestworker_") and f.endswith(".txt")]
    for f in log_files:
        os.remove(os.path.join(t_d, f))
    return

def plot_publication():
    md = "master_dsi_standard_easy"
    dsi = DSI.load(os.path.join(md,"dsi.pickle"))
    data = dsi.data.copy()
    pst = pyemu.Pst(os.path.join(md,"dsi.pst"))
    obs = pst.observation_data
    ptoe = pst.ies.obsen3.copy()
    proe = pst.ies.obsen0.copy()
    forecasts  = [o for o in obs.obgnme.unique() if 
                any([f in o for f in 
                    ['trgw-0-9-1','gage-1','tailwater']])
                    ]
    forecasts
    title_dict = {'trgw-0-9-1':'groundwater level','gage-1':'streamflow','tailwater':'sw-gw exchange'}


    # standard DSI figure
    fig,axs = plt.subplots(len(forecasts),1,figsize=(3.24,6.75),sharex=True)

    for i,fc in enumerate(forecasts):
        ax = axs[i]
        char = chr(97 + i)  # 97 is the ASCII code for 'a'
        ax.set_title(f"({char}) {title_dict[fc.split(':')[-1]]}", fontsize=10, loc='left')

        df = obs.loc[obs.obgnme==fc].copy()
        df.time = df.time.astype(float)
        times = df.time.unique() - df.time.min()
        times = sorted(times)
        df.sort_values(by='time',inplace=True)
        [ax.plot(times,data.loc[i,df.obsnme.tolist()],'-',color='orange',alpha=0.3,label="fom-prior") for i in data.index]

        [ax.plot(times,proe.loc[i,df.obsnme.tolist()],'-',color='0.5',alpha=0.3,label="dsi-prior") for i in proe.index]
        [ax.plot(times,ptoe.loc[i,df.obsnme.tolist()],'-',color='b',alpha=0.3,label="dsi-posterior") for i in ptoe.index]


        ax.plot(times,df.obsval,'-',color='red',label="truth")
        #ax.plot(times,proe.loc["base",df.obsnme.tolist()],'-',color='blue',label="prior")

        nz = df.loc[df.weight>0,:]
        if nz.shape[0]>0:
            ax.plot(nz.time - df.time.min(),nz.obsval,'x',color='black',label="measured")

    ax.set_xlim(times[0],times[-1])
    ax.set_xlabel("time (days)")
    # small tick labels
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=8)
        

    # remove duplicate labels
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[-1].legend(by_label.values(), by_label.keys(),fontsize=8,loc='lower right',)

    axs[0].set_ylabel("$(m)$")
    axs[1].set_ylabel("$(ML/d)$")
    axs[2].set_ylabel("$(ML/d)$")

    for ax in axs[1:]:
        # divide ytick lables by 1000
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y/1000:.1f}" for y in yticks])

    fig.tight_layout()
    fig.savefig("dsi_forecasts.pdf",dpi=600)


    ## pater-n-DSI figure
    md = "master_dsi_standard"
    pst = pyemu.Pst(os.path.join(md,"dsi.pst"))
    obs = pst.observation_data
    ptoe = pst.ies.obsen3.copy()
    proe = pst.ies.obsen0.copy()


    dsi = DSI.load(os.path.join(md,"dsi.pickle"))
    data = dsi.data.copy()


    pst_p = pyemu.Pst(os.path.join("master_dsi_row","dsi.pst"))
    #obs = pst.observation_data
    row_ptoe = pst_p.ies.obsen3.copy()
    row_proe = pst_p.ies.obsen0.copy()

    forecasts  = [o for o in obs.obgnme.unique() if 
                any([f in o for f in 
                    ['trgw-0-9-1','gage-1','tailwater']])
                    ]
    forecasts
    title_dict = {'trgw-0-9-1':'groundwater level','gage-1':'streamflow','tailwater':'sw-gw exchange'}

    fig,axs = plt.subplots(len(forecasts),2,figsize=(6.75,6.75),sharex=True)
    counter=0
    for i,fc in enumerate(forecasts):
        df = obs.loc[obs.obgnme==fc].copy()
        df.time = df.time.astype(float)
        times = df.time.unique() - df.time.min()
        times = sorted(times)
        df.sort_values(by='time',inplace=True)

        for j,ax in enumerate(axs[i,:]):
            
            if j>0:
                tag = "pattern-dsi"
                ptoe_use = row_ptoe
                proe_use = row_proe
            else:
                tag = "dsi"
                ptoe_use = ptoe
                proe_use = proe

            char = chr(97 + counter)  # 97 is the ASCII code for 'a'
            counter+=1
            ax.set_title(f"({char}) {title_dict[fc.split(':')[-1]]}: {tag}", fontsize=10, loc='left')

            [ax.plot(times,data.loc[i,df.obsnme.tolist()],'-',color='orange',alpha=0.3,label="fom-prior") for i in data.index]
            [ax.plot(times,proe_use.loc[i,df.obsnme.tolist()],'-',color='0.5',alpha=0.3,label="dsi-prior") for i in proe_use.index]
            [ax.plot(times,ptoe_use.loc[i,df.obsnme.tolist()],'-',color='b',alpha=0.3,label="dsi-posterior") for i in ptoe_use.index]

            ax.plot(times,df.obsval,'-',color='red',label="truth")
            #ax.plot(times,proe.loc["base",df.obsnme.tolist()],'-',color='blue',label="prior")

            nz = df.loc[df.weight>0,:]
            if nz.shape[0]>0:
                ax.plot(nz.time - df.time.min(),nz.obsval,'x',color='black',label="measured")

    ax.set_xlim(times[0],times[-1])
    for ax in axs[-1,:]:
        ax.set_xlabel("time (days)")
    # small tick labels
    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize=8)
        

    # remove duplicate labels
    handles, labels = axs[0,0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0,1].legend(by_label.values(), by_label.keys(),fontsize=8,loc='lower left')

    axs[0,0].set_ylabel("$(m)$")
    axs[1,0].set_ylabel("$(ML/d)$")
    axs[2,0].set_ylabel("$(ML/d)$")

    for row in axs[1:]:

        row[0].set_ylim(row[1].get_ylim())

        # divide ytick lables by 1000
        yticks = row[0].get_yticks()
        row[0].set_yticklabels([f"{y/1000:.1f}" for y in yticks])
        row[1].set_yticklabels([])#f"{y/1000:.1f}" for y in yticks])
    axs[0,1].set_yticklabels([])#f"{y/1000:.1f}" for y in yticks])



    fig.tight_layout()
    fig.savefig("dsi_pattern_forecasts.pdf",dpi=600)


    return


def plot_rowwise_scaling():

    # setup synthetic data
    t = np.linspace(0, 20, 100)
    # A simple exponential decay + offset
    def model_response(t, A, k, C):
        return A * (1 - np.exp(-k * t)) + C
    # Represents field data with specific magnitude
    truth_params = {'A': 5.0, 'k': 0.2, 'C': 2.0}
    y_truth = model_response(t, **truth_params)
    # Represents structural error: consistently estimating higher heads/drawdowns
    n_reals = 20
    prior_ensemble = []
    np.random.seed(42)

    for _ in range(n_reals):
        # Same 'k' (shape) but biased 'A' and 'C' (magnitude)
        A_sim = np.random.uniform(12, 18)
        k_sim = np.random.normal(0.2, 0.02) # Slight variation in shape, but mostly correct
        C_sim = np.random.uniform(8, 12)
        
        y_sim = model_response(t, A_sim, k_sim, C_sim)
        prior_ensemble.append(y_sim)

    prior_ensemble = np.array(prior_ensemble)

    # Apply Row-Wise Normalization
    def row_wise_normalize(data):
        # Data shape: (n_samples, n_timesteps)
        mins = np.min(data, axis=1, keepdims=True)
        maxs = np.max(data, axis=1, keepdims=True)
        return (data - mins) / (maxs - mins + 1e-10) # epsilon for stability

    # Normalize prior
    prior_norm = row_wise_normalize(prior_ensemble)

    # Normalize truth (reshape to 1, n_times for broadcasting)
    truth_norm = row_wise_normalize(y_truth.reshape(1, -1))

    # make figure
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3),sharex=True)

    # Real Space (Conflict)
    ax = axes[0]
    for i in range(n_reals):
        ax.plot(t, prior_ensemble[i], color='gray', alpha=0.5, linewidth=1, label='Prior Ensemble' if i==0 else "")
    ax.plot(t, y_truth, color='red', linewidth=2.5, label='Measured')
    ax.set_title("(a) Raw data", fontsize=10, loc='left')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Value (-)", fontsize=10)
    # remove duplicate labels
    #handles, labels = ax.get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper left')

    ax.grid(True, linestyle='--', alpha=0.6)
    #ax.text(0.05, 0.95, "Structural Bias:\nPrior range does not\ncover observation", 
    #        transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))

    # Pattern Space (Resolved)
    ax = axes[1]
    for i in range(n_reals):
        ax.plot(t, prior_norm[i], color='0.5', alpha=0.5, linewidth=1, label='Prior Ensemble' if i==0 else "")
    ax.plot(t, truth_norm[0], color='red', linewidth=2.5, label='Measured')
    ax.set_title("(b) Row-wise normalized data", fontsize=10, loc='left')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Normalized value [0,1]", fontsize=10)

    # remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='lower right')


    ax.grid(True, linestyle='--', alpha=0.6)
    #ax.text(0.05, 0.95, "Pattern Matching:\nShapes align despite\nmagnitude error", 
    #        transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))

    for ax in axes:
        # ticklabel size
        ax.tick_params(axis='both', which='major', labelsize=8)

    fig.tight_layout()
    plt.savefig('rowwisenormalized.pdf', dpi=600)
    return


if __name__ == "__main__":

    run_freyberg =              True
    run_convenient_truth =      True
    run_inconvenient_truth =    True

    if run_freyberg:
        run_training_ensemble()

    if run_convenient_truth:
        oe,pst = load_training_data('master_prior')
        drop_obs(pst,"master_prior")
        #truth_real, sorted_index = choose_inconvenient_truth(pst,oe,
        #                                                    usecol="trgw-0-13-10",#"gage-1",
        #                                                    ascending=True,
        #                                                    )
        _data, obsdata, well_input_cols = add_wellpars_to_data(pst,oe)
        data = _data.loc[:,pst.obs_names].sample(n=200,random_state=0).copy()
        obsdata = update_obsdata_with_truth(obsdata,"base",oe)
        
        if "base" not in data.index:
            # rename last index base
            data.rename(index={data.index[-1]:"base"},inplace=True)
            data.loc["base",:] = _data.loc["base",pst.obs_names].values
        #groups, fit_groups = get_fitbygroups(pst)

        obsnmes = obsdata.loc[obsdata.usecol=="gage-1"].obsnme.tolist()
        transforms = [{"type":"log10","columns":obsnmes},
                      {"type":"normal_score"}]

        dsi,pst = prepare_dsi_dir(obsdata,data,
                                groups=None,
                                fit_groups=None,
                                transforms=transforms)
        run_dsi(tag="standard_easy")  


        oe_dict= {}
        pst = pyemu.Pst(os.path.join("master_dsi_standard_easy","dsi.pst"))
        oe_dict["b"] = pst.ies.obsen3.copy()
        figures = plot_oe_tseries(pst,oe_dict,)
        # save as pdf
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('results_easy.pdf') as pdf:
            # plot each figure
            for f in figures:
                pdf.savefig(f)
            pdf.close()  


    if run_inconvenient_truth:
        oe,pst = load_training_data('master_prior')
        drop_obs(pst,"master_prior")
        truth_real, sorted_index = choose_inconvenient_truth(pst,oe,
                                                            usecol="trgw-0-13-10",#"gage-1",
                                                            ascending=True,
                                                            )
        data, obsdata, well_input_cols = add_wellpars_to_data(pst,oe)
        obsdata = update_obsdata_with_truth(obsdata,truth_real,oe)

        nreals = 200#3 *(len(sorted_index)//5)
        data = data.loc[sorted_index[:nreals],pst.obs_names].copy()
        groups, fit_groups = get_fitbygroups(pst)

        obsnmes = obsdata.loc[obsdata.usecol=="gage-1"].obsnme.tolist()
        transforms = [{"type":"log10","columns":obsnmes}]

        dsi,pst = prepare_dsi_dir(obsdata,data,
                                groups=None,
                                fit_groups=None,
                                transforms=transforms)
        run_dsi(tag="standard")

        dsi,pst = prepare_dsi_dir(obsdata,data,
                                groups,fit_groups,
                                transforms=None)
        run_dsi(tag="row")

        plot_results()


    plot_publication()
    plot_rowwise_scaling()
    