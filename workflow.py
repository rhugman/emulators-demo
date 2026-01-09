import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyemu
import platform
from pyemu.emulators import GPR

# set random seed
np.random.seed(42)

# set path to pestpp executables
bin_path = os.path.join("bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path, "linux")
elif "macos" in platform.platform().lower():
    bin_path = os.path.join(bin_path, "mac")
else:
    bin_path = os.path.join(bin_path, "win")

exe = ""
if "windows" in platform.platform().lower():
    exe = ".exe"
ies_path = os.path.abspath(os.path.join(bin_path, "pestpp-ies" + exe))
mou_path = os.path.abspath(os.path.join(bin_path, "pestpp-mou" + exe))
if not os.path.exists(ies_path):
    pyemu.utils.get_pestpp(bin_path)
    assert os.path.exists(ies_path), "pestpp-ies not found"
if not os.path.exists(mou_path):
    pyemu.utils.get_pestpp(bin_path)
assert os.path.exists(mou_path), "pestpp-mou not found"


def map_hosaki(t_d):
    pst = pyemu.Pst(os.path.join(t_d,"pest.pst"))
    par = pst.parameter_data
    pvals = []
    sweep_steps = 30
    for p1 in np.linspace(par.parlbnd.iloc[0],par.parubnd.iloc[0],sweep_steps):
        for p2 in np.linspace(par.parlbnd.iloc[0],par.parubnd.iloc[0],sweep_steps):
            pvals.append([p1,p2])
    pvals = pd.DataFrame(pvals,columns=pst.par_names)
    pvals.to_csv(os.path.join(t_d,"sweep_in.csv"))
    pst.pestpp_options["ies_par_en"] = "sweep_in.csv"
    pst.control_data.noptmax = -1
    sweep_d = os.path.join("hosaki_sweep")
    pst.pestpp_options["ies_include_base"] = False

    pst.write(os.path.join(t_d,"pest.pst"))
    port = 5544
    num_workers = 10
    sys.path.insert(0,t_d)
    from forward_run import hosaki_ppw_worker as ppw_function
    pyemu.os_utils.start_workers(t_d,ies_path,"pest.pst",
                                num_workers=num_workers,
                                master_dir=sweep_d,worker_root='.',
                                port=port,ppw_function=ppw_function)
        
    sweep_x,sweep_y,sweep_z = load_hosaki_sweep_data(sweep_d)
    return sweep_x,sweep_y,sweep_z

def load_hosaki_sweep_data(sweep_d):
    pst = pyemu.Pst(os.path.join("hosaki_template","pest.pst"))
    sweep_pe = pd.read_csv(os.path.join(sweep_d,"pest.0.par.csv"),index_col=0)
    sweep_oe = pd.read_csv(os.path.join(sweep_d,"pest.0.obs.csv"),index_col=0)
    sweep_steps = int(np.sqrt(sweep_pe.shape[0]))
    sweep_x = sweep_pe.loc[:,pst.par_names[0]].values.reshape(sweep_steps,sweep_steps)
    sweep_y = sweep_pe.loc[:,pst.par_names[1]].values.reshape(sweep_steps,sweep_steps)
    sweep_z = sweep_oe.loc[:,pst.obs_names[0]].values.reshape(sweep_steps,sweep_steps)

    return sweep_x,sweep_y,sweep_z

def get_obj_map(ax,_sweep_x,_sweep_y,_sweep_z,label="objective function",
                levels=[-2,-1,0,0.5],vmin=-2,vmax=0.5,cmap="magma"): 
    """a simple function to plot an objective function surface"""
    cb = ax.pcolormesh(_sweep_x,_sweep_y,_sweep_z,vmin=vmin,vmax=vmax,cmap=cmap)
    cbar = plt.colorbar(cb,ax=ax)
    cbar.ax.tick_params(labelsize=8)
    ax.contour(_sweep_x,_sweep_y,_sweep_z,levels=levels,colors="w")
    
    ax.set_aspect("equal")
    return ax

def plot_obj_map(_sweep_x,_sweep_y,_sweep_z,label="objective function",
                 levels=[-2,-1,0,0.5],vmin=-2,vmax=0.5,cmap="magma"):
    fig,ax = plt.subplots(1,1,figsize=(6,5))
    _ = get_obj_map(ax,_sweep_x,_sweep_y,_sweep_z,label=label,
                    levels=levels,vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_title("truth",loc="left")
    #plt.show()
    fig.savefig("hosaki_objective_function_truth.png",dpi=300)
    plt.close(fig)


def run_hosaki_process_mou(t_d,m_d="hosaki_model_master",popsize=10):
    sys.path.insert(0,t_d)
    from forward_run import hosaki_ppw_worker as ppw_function
    port = 5544
    pst = pyemu.Pst(os.path.join(t_d,"pest.pst"))
    par = pst.parameter_data
    
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    pst.pestpp_options["mou_population_size"] = popsize
    pst.control_data.noptmax = -1
    par = pst.parameter_data
    par.loc[pst.par_names[0],"parubnd"] = 3
    par.loc[pst.par_names[0],"parval1"] = 1.5
    pst.write(os.path.join(t_d,"pest.pst"))
    num_workers = 1
    pyemu.os_utils.start_workers(t_d,mou_path,"pest.pst",
                                    num_workers=num_workers,
                                    master_dir=m_d,worker_root='.',
                                    port=port,ppw_function=ppw_function)
    return

def load_hosaki_training_data(m_d):
    pst = pyemu.Pst(os.path.join("hosaki_template","pest.pst"))
    training_dvpop_fname = os.path.join(m_d,"pest.0.dv_pop.csv")
    training_opop_fname = os.path.join(m_d,"pest.0.obs_pop.csv")
    training_dvpop = pd.read_csv(training_dvpop_fname,index_col=0)
    training_opop = pd.read_csv(training_opop_fname,index_col=0)

    return pst,training_dvpop,training_opop

def plot_hosaki_objfunc_and_training_points(m_d,sweep_x,sweep_y,sweep_z,_data=None):
    iiter = m_d.split("_")[-1]


    pst, training_dvpop, training_opop = load_hosaki_training_data(m_d)
    if _data is not None:
        training_dvpop = _data
    
    fig,ax = plt.subplots(1,1,figsize=(6,5))
    get_obj_map(ax,sweep_x,sweep_y,sweep_z)
    ax.scatter(training_dvpop.loc[:,pst.par_names[0]],training_dvpop.loc[:,pst.par_names[1]],marker='^',c='w',s=20)
    ax.set_title("true objective function surface with training points")
    #plt.show()
    fig.savefig(f"hosaki_objective_function_with_training_points_{iiter}.png",dpi=300)
    plt.close(fig)

    return

def prepare_gpr_emulator(data, input_names, output_names, pst_dir, case='pest', gpr_t_d='template_gpr',noptmax=-1):
    
    gpr = GPR(data=data.copy(),
          input_names=input_names,
          output_names=output_names,
          #transforms=transforms,
          #kernel=gp_kernel,
          n_restarts_optimizer=20,
          )
    gpr.fit()
    gpr.prepare_pestpp(pst_dir,case,gpr_t_d=gpr_t_d,mou_path=mou_path)

    gpst = pyemu.Pst(os.path.join(gpr_t_d,"pest_gpr.pst"))
    # some bits and bobs:
    gpst.control_data.noptmax = noptmax
    gpst.pestpp_options["ies_include_base"] = False
    gpst.pestpp_options["mou_save_population_every"] = 1
    gpst.pestpp_options.pop("mou_dv_population_file",None)
    par = gpst.parameter_data
    par.loc[:,"parlbnd"] = 0.0
    par.loc[:,"parubnd"] = 5.0
    par.loc[:,"parval1"] = 2.5    
    
    gpst.write(os.path.join(gpr_t_d,"pest_gpr.pst"),version=2)
    return gpr, gpst

def run_gpr_mou(gpr_t_d,noptmax=8,inipop=None):
    gpr = GPR.load(os.path.join(gpr_t_d,"gpr_emulator.pkl"))
    input_df = pd.read_csv(os.path.join(gpr_t_d,"gpr_input.csv"),index_col=0)
    gpst = pyemu.Pst(os.path.join(gpr_t_d,"pest_gpr.pst"))
    gpst.control_data.noptmax = noptmax
    gpst.pestpp_options["mou_population_size"] = 40
    if inipop is not None:
        inipop = inipop.loc[:,gpst.par_names]

    else:
        inipop = pyemu.ParameterEnsemble.from_uniform_draw(gpst,gpst.pestpp_options["mou_population_size"])
    inipop.to_csv(os.path.join(gpr_t_d,"inipop.csv"))
    gpst.pestpp_options["mou_dv_population_file"] = "inipop.csv"
    gpst.write(os.path.join(gpr_t_d,"pest_gpr.pst"))
    gpr_m_d = gpr_t_d.replace("template","master")
    num_workers = 10
    pyemu.os_utils.start_workers(gpr_t_d,mou_path,"pest_gpr.pst",
                                num_workers=num_workers,
                                master_dir=gpr_m_d,worker_root='.',
                                ppw_function=pyemu.helpers.gpr_pyworker,
                                    ppw_kwargs={"input_df":input_df,
                                                "gpr":gpr})

    return

def run_gpr_sweep(gpr,gpr_t_d):
    shutil.copy2(os.path.join("hosaki_sweep","sweep_in.csv"),os.path.join(gpr_t_d,"sweep_in.csv"))
    #gpr = GPR.load(os.path.join(gpr_t_d,"gpr_emulator.pkl"))
    input_df = pd.read_csv(os.path.join(gpr_t_d,"gpr_input.csv"),index_col=0)
    gpr_sweep_d = gpr_t_d.replace("template","sweep")
    num_workers = 10
    pyemu.os_utils.start_workers(gpr_t_d,
                                 ies_path,"pest_gpr.pst", num_workers=num_workers, worker_root=".",#port=5544,
                                 master_dir=gpr_sweep_d, verbose=True, 
                                 ppw_function=pyemu.helpers.gpr_pyworker,
                                 ppw_kwargs={"input_df":input_df,
                                            "gpr":gpr})

    return

def plot_gpr_sweep_results(_m_d,_gpr_sweep_d,sweep_steps = 30,_data=None):

    pst, _training_dvpop, training_opop = load_hosaki_training_data(_m_d)
    if _data is not None:
        _training_dvpop = _data
    iiter = _gpr_sweep_d.split("_")[-1]
    gpst = pyemu.Pst(os.path.join(_gpr_sweep_d,"pest_gpr.pst"))
    #load the gpr sweep results to viz the emulated objective function surface
    sweep_gpr_pe = pd.read_csv(os.path.join(_gpr_sweep_d,"pest_gpr.0.par.csv"),index_col=0)
    sweep_gpr_oe = pd.read_csv(os.path.join(_gpr_sweep_d,"pest_gpr.0.obs.csv"),index_col=0)
    gpr_sweep_x = sweep_gpr_pe.loc[:,gpst.par_names[0]].values.reshape(sweep_steps,sweep_steps)
    gpr_sweep_y = sweep_gpr_pe.loc[:,gpst.par_names[1]].values.reshape(sweep_steps,sweep_steps)
    gpr_sweep_z = sweep_gpr_oe.loc[:,gpst.obs_names[0]].values.reshape(sweep_steps,sweep_steps)
    gpr_sweep_stdev_z = sweep_gpr_oe.loc[:,gpst.obs_names[1]].values.reshape(sweep_steps,sweep_steps)
    
    # plot it up:
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    axes = axes.flatten()
    pst = pyemu.Pst(os.path.join("hosaki_template","pest.pst"))
    sweep_x,sweep_y,sweep_z = load_hosaki_sweep_data("hosaki_sweep")
    get_obj_map(axes[0],sweep_x,sweep_y,sweep_z)
    get_obj_map(axes[1],gpr_sweep_x,gpr_sweep_y,gpr_sweep_z)
    axes[1].scatter(_training_dvpop.loc[:,pst.par_names[0]],_training_dvpop.loc[:,pst.par_names[1]],marker='^',c='w',s=20)
    diff = sweep_z-gpr_sweep_z
    amax = np.abs(diff).max()
    get_obj_map(axes[2],gpr_sweep_x,gpr_sweep_y,sweep_z-gpr_sweep_z,label="truth minus emulated",levels=None,vmin=-amax,vmax=amax,cmap="bwr")
    get_obj_map(axes[3],gpr_sweep_x,gpr_sweep_y,gpr_sweep_stdev_z,label="GPR stdev",levels=None,
                vmin=gpr_sweep_stdev_z.min(),vmax=gpr_sweep_stdev_z.max(),cmap="jet")
    axes[2].scatter(_training_dvpop.loc[:,pst.par_names[0]],_training_dvpop.loc[:,pst.par_names[1]],marker='^',c='w',s=20)
    
    axes[0].set_title("truth",loc="left")
    axes[1].set_title("emulated with training points",loc="left")
    axes[2].set_title("difference with training points",loc="left")
    axes[3].set_title("GPR standard deviation",loc="left")
    #plt.show()
    fig.savefig(f"hosaki_gpr_emulator_sweep_results_{iiter}.png",dpi=300)
    plt.close(fig)
    return


def plot_gpr_mou_results(_gpr_m_d,_gpr_sweep_d,sweep_steps=30):
    iiter = _gpr_m_d.split("_")[-1]

    gpst = pyemu.Pst(os.path.join(_gpr_sweep_d,"pest_gpr.pst"))
    #load the true hosaki function over the sweep grid
    sweep_x,sweep_y,sweep_z = load_hosaki_sweep_data("hosaki_sweep")
    #load the gpr sweep results to viz the emulated objective function surface  

    sweep_gpr_pe = pd.read_csv(os.path.join(_gpr_sweep_d,"pest_gpr.0.par.csv"),index_col=0)
    sweep_gpr_oe = pd.read_csv(os.path.join(_gpr_sweep_d,"pest_gpr.0.obs.csv"),index_col=0)
    gpr_sweep_x = sweep_gpr_pe.loc[:,gpst.par_names[0]].values.reshape(sweep_steps,sweep_steps)
    gpr_sweep_y = sweep_gpr_pe.loc[:,gpst.par_names[1]].values.reshape(sweep_steps,sweep_steps)
    gpr_sweep_z = sweep_gpr_oe.loc[:,gpst.obs_names[0]].values.reshape(sweep_steps,sweep_steps)
    gpr_sweep_stdev_z = sweep_gpr_oe.loc[:,gpst.obs_names[1]].values.reshape(sweep_steps,sweep_steps)
    
    gpr_dvpops = [os.path.join(_gpr_m_d,f) for f in os.listdir(_gpr_m_d) if len(f.split('.')) == 4 and f.endswith("dv_pop.csv") and "archive" not in f]
    gpr_dvpops_itr = [int(f.split(".")[1]) for f in gpr_dvpops]
    gpr_dvpops = {itr:pd.read_csv(f,index_col=0) for itr,f in zip(gpr_dvpops_itr,gpr_dvpops)}
    for itr in range(max(gpr_dvpops_itr)):
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        ax = axes[0]
        get_obj_map(ax,sweep_x,sweep_y,sweep_z)
        ax.scatter(gpr_dvpops[itr].loc[:,gpst.par_names[0]],gpr_dvpops[itr].loc[:,gpst.par_names[1]],marker='.',c='w',s=10)
        ax.set_title("truth generation {0}".format(itr),loc="left")
        ax = axes[1]
        get_obj_map(ax,gpr_sweep_x,gpr_sweep_y,gpr_sweep_z)
        ax.scatter(gpr_dvpops[itr].loc[:,gpst.par_names[0]],gpr_dvpops[itr].loc[:,gpst.par_names[1]],marker='.',c='w',s=10)
        ax.set_title("emulated generation {0}".format(itr),loc="left")
        #plt.show()
        fig.savefig(f"hosaki_gpr_emulator_mou_generation_{iiter}_{itr:03d}.png",dpi=300)
        plt.close(fig)


def fillin_hosaki_template(t_d,gpr_m_d,iiter):
    pst = pyemu.Pst(os.path.join(t_d,"pest.pst"))
    gpst = pyemu.Pst(os.path.join(gpr_m_d,"pest_gpr.pst"))

    gpr_dvpops = [os.path.join(gpr_m_d,f) for f in os.listdir(gpr_m_d) if len(f.split('.')) == 4 and f.endswith("dv_pop.csv") and "archive" not in f]
    gpr_dvpops_itr = [int(f.split(".")[1]) for f in gpr_dvpops]
    gpr_dvpops = {itr:pd.read_csv(f,index_col=0) for itr,f in zip(gpr_dvpops_itr,gpr_dvpops)}
    gpr_dvpops[max(gpr_dvpops_itr)].iloc[:10].to_csv(os.path.join(t_d,f"retrain_{iiter}_dvpop.csv"))
    
    pst.pestpp_options["mou_dv_population_file"] = f"retrain_{iiter}_dvpop.csv"
    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d,"pest.pst"))

    return

def train_gpr(gpr_t_d,m_d,data=None):
    # load the training data
    pst, training_dvpop, training_opop = load_hosaki_training_data(m_d)
    _data = training_dvpop.join(training_opop).copy()
    if data is None:
        data = _data
    elif isinstance(data,pd.DataFrame):
        data = pd.concat([data,_data],axis=0)
        print("training data size:",data.shape)
    input_names = training_dvpop.columns.tolist()
    output_names = training_opop.columns.tolist()
    gpr,gpst = prepare_gpr_emulator(data, input_names, output_names,pst_dir=m_d,gpr_t_d=gpr_t_d)
    assert gpr is not None
    assert isinstance(gpr,GPR)
    return gpr, data

def hosaki_gpr_demo():

    # copy tempalte dir
    org_d = os.path.join("templates","hosaki_template")
    assert os.path.exists(org_d)
    t_d = "hosaki_template"
    if os.path.exists(t_d):
        shutil.rmtree(t_d)
    shutil.copytree(org_d,t_d)


    # map the true hosaki function over a grid of param values
    sweep_x,sweep_y,sweep_z = map_hosaki(t_d)
    plot_obj_map(sweep_x,sweep_y,sweep_z,label="objective function",
                 levels=[-2,-1,0,0.5],vmin=-2,vmax=0.5,cmap="magma")

    # run the process model once to get the sample of initial population
    m_d = "hosaki_model_master"
    popsize = 10

    run_hosaki_process_mou(t_d,m_d=m_d+ f"_0",popsize=popsize)
    

    data = None
    inipop = None
    _noptmax = 8
    for iiter in range(5):
        _m_d = m_d+ f"_{iiter}"
        _gpr_t_d = f"template_gpr_{iiter}"
        _gpr_m_d = _gpr_t_d.replace("template","master")
        _sweep_md = _gpr_t_d.replace("template","sweep")

    
        gpr, data = train_gpr(_gpr_t_d,_m_d, data=data)

        # run sweep with gpr emulator
        run_gpr_sweep(gpr,gpr_t_d = _gpr_t_d)

        if iiter >= 4:
            inipop = pd.read_csv(os.path.join(f"master_gpr_{iiter-1}",f"pest_gpr.{_noptmax}.dv_pop.csv"),index_col=0)
            _noptmax = 20
        run_gpr_mou(gpr_t_d=_gpr_t_d,noptmax=_noptmax,inipop=inipop)

        fillin_hosaki_template(t_d,_gpr_m_d,iiter=iiter)
        run_hosaki_process_mou(t_d,m_d=m_d+ f"_{iiter+1}",popsize=popsize)

        plot_hosaki_objfunc_and_training_points(_m_d,sweep_x,sweep_y,sweep_z)
        plot_gpr_sweep_results(_m_d,_sweep_md,sweep_steps=30,_data=data)
        plot_gpr_mou_results(_gpr_m_d,_sweep_md,sweep_steps=30)


    print(data)
    return


def plot_pub():
    def load_gpr_sweep_data(_gpr_sweep_d,sweep_steps=30):
        gpst = pyemu.Pst(os.path.join(_gpr_sweep_d,"pest_gpr.pst"))
        #load the gpr sweep results to viz the emulated objective function surface
        sweep_gpr_pe = pd.read_csv(os.path.join(_gpr_sweep_d,"pest_gpr.0.par.csv"),index_col=0)
        sweep_gpr_oe = pd.read_csv(os.path.join(_gpr_sweep_d,"pest_gpr.0.obs.csv"),index_col=0)
        gpr_sweep_x = sweep_gpr_pe.loc[:,gpst.par_names[0]].values.reshape(sweep_steps,sweep_steps)
        gpr_sweep_y = sweep_gpr_pe.loc[:,gpst.par_names[1]].values.reshape(sweep_steps,sweep_steps)
        gpr_sweep_z = sweep_gpr_oe.loc[:,gpst.obs_names[0]].values.reshape(sweep_steps,sweep_steps)
        gpr_sweep_stdev_z = sweep_gpr_oe.loc[:,gpst.obs_names[1]].values.reshape(sweep_steps,sweep_steps)
        return gpr_sweep_x, gpr_sweep_y, gpr_sweep_z, gpr_sweep_stdev_z

    iiters = [i.split("_")[-1] for i in os.listdir() if i.startswith("master_gpr_")]
    iiters.sort()

    fig,axs = plt.subplots(2,3,figsize=(7,4),sharex=True,sharey=True)
    axs = axs.flatten()

    pst = pyemu.Pst(os.path.join("hosaki_template","pest.pst"))
    par = pst.parameter_data
    sweep_x,sweep_y,sweep_z = load_hosaki_sweep_data("hosaki_sweep")
    get_obj_map(axs[0],sweep_x,sweep_y,sweep_z)
    axs[0].set_title("Truth",size=10)

    data_dict = {}
    data = None
    m_d = "hosaki_model_master"
    for iiter in iiters:
        ax = axs[int(iiter)+1]
        
        
        _m_d = m_d+ f"_{iiter}"
        _gpr_t_d = f"template_gpr_{iiter}"
        _gpr_m_d = _gpr_t_d.replace("template","master")
        _sweep_md = _gpr_t_d.replace("template","sweep")

        # load the training data
        pst, training_dvpop, training_opop = load_hosaki_training_data(_m_d)
        _data = training_dvpop.join(training_opop).copy()
        if data is None:
            data = _data
        elif isinstance(data,pd.DataFrame):
            data = pd.concat([data,_data],axis=0)
            print("training data size:",data.shape)
        gpr_sweep_x, gpr_sweep_y, gpr_sweep_z, gpr_sweep_stdev_z = load_gpr_sweep_data(_sweep_md)
        get_obj_map(ax,gpr_sweep_x,gpr_sweep_y,gpr_sweep_z)
        
        ax.scatter(data.loc[:,pst.par_names[0]],data.loc[:,pst.par_names[1]],marker='.',c='w',s=10)

        fnames = [i for i in os.listdir(_gpr_m_d) if i.endswith(".dv_pop.csv") and ".archive." not in i and i.count(".")==3]
        fnames.sort(key=lambda x: int(x.split(".")[1]))
        dvopt = pd.read_csv(os.path.join(_gpr_m_d,fnames[-1]),index_col=0)
        ax.scatter(dvopt.loc[:,pst.par_names[0]],dvopt.loc[:,pst.par_names[1]],marker='.',c='r',s=10)
        
        ax.set_title(f"Iteration {iiter}",size=10)

    for ax in axs:
        ax.set_xlabel(pst.par_names[0],size=8)
        ax.set_ylabel(pst.par_names[1],size=8)
        ax.grid()

    fig.tight_layout()
    fig.savefig("hosaki_gpr_emulator_figure.png",dpi=300)
    return

if __name__ == "__main__":
    #hosaki_gpr_demo()
    plot_pub()