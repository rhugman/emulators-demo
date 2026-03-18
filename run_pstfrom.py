import sys
import os
import shutil
import platform
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
# for this course we use locally stored version of pyemu and flopy to avoid version conflicts
import pyemu
import flopy


from workflow_dsi import get_bins
#import herebedragons as hbd

import os
import numpy as np



def prep_forecasts(pst, model_times=False):
    pred_csv = os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth',"pred_data.csv")
    assert os.path.exists(pred_csv)
    pred_data = pd.read_csv(pred_csv)
    pred_data.set_index('site', inplace=True)
    
    if type(model_times) == bool:
        model_times = [float(i) for i in pst.observation_data.time.unique()]
        
    ess_obs_data = {}
    for site in pred_data.index.unique().values:
        site_obs_data = pred_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times,method="nearest")
        #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
        ess_obs_data[site] = sm_site_obs_data
    obs_data = pd.DataFrame(ess_obs_data)

    obs = pst.observation_data
    obs_names = [o for o in pst.obs_names if o not in pst.nnz_obs_names]

    # get list of times for obs name suffixes
    time_str = obs_data.index.map(lambda x: f"time:{x}").values
    # empyt list to keep track of missing observation names
    missing=[]
    for col in obs_data.columns:
        if col.lower()=='part_time':
            obs_sufix = col.lower()
        else:
        # get obs list suffix for each column of data
            obs_sufix = col.lower()+"_"+time_str
        if type(obs_sufix)==str:
            obs_sufix=[obs_sufix]

        for string, oval, time in zip(obs_sufix,obs_data.loc[:,col].values, obs_data.index.values):
                if not any(string in obsnme for obsnme in obs_names):
                    missing.append(string)
                # if not, then update the pst.observation_data
                else:
                    # get a list of obsnames
                    obsnme = [ks for ks in obs_names if string in ks] 
                    if type(obsnme) == str:
                        obsnme=[obsnme]
                    obsnme = obsnme[0]
                    if obsnme=='part_time':
                        oval = pred_data.loc['part_time', 'value']
                    # assign the obsvals
                    obs.loc[obsnme,"obsval"] = oval
                        ## assign a generic weight
                        #if time > 3652.5 and time <=4018.5:
                        #    obs.loc[obsnme,"weight"] = 1.0      
    return 

def process_secondary_obs(ws='.'):
    # load dependencies inside the function so that they get carried over to forward_run.py by PstFrom
    import os
    import pandas as pd

    def write_tdif_obs(orgf, newf, ws='.'):
        df = pd.read_csv(os.path.join(ws,orgf), index_col='time')
        df = df - df.iloc[0, :]
        df.to_csv(os.path.join(ws,newf))
        return

    # write the tdiff observation csv's
    write_tdif_obs('heads.csv', 'heads.tdiff.csv', ws)
    write_tdif_obs('sfr.csv', 'sfr.tdiff.csv', ws)

    print('Secondary observation files processed.')
    return


def extract_hds_arrays_and_list_dfs():
    import flopy
    hds = flopy.utils.HeadFile("freyberg6_freyberg.hds")
    for it,t in enumerate(hds.get_times()):
        d = hds.get_data(totim=t)
        for k,dlay in enumerate(d):
            np.savetxt("hdslay{0}_t{1}.txt".format(k+1,it+1),d[k,:,:],fmt="%15.6E")

    lst = flopy.utils.Mf6ListBudget("freyberg6.lst")
    inc,cum = lst.get_dataframes(diff=True,start_datetime=None)
    inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
    cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
    inc.index.name = "totim"
    cum.index.name = "totim"
    inc.to_csv("inc.csv")
    cum.to_csv("cum.csv")
    return


def test_extract_hds_arrays(d):
    cwd = os.getcwd()
    os.chdir(d)
    extract_hds_arrays_and_list_dfs()
    os.chdir(cwd)




def run_pstfrom(template_ws):
        
    # %% [markdown]
    # We will be calling a few external programs throughout this tutorial. Namely, MODFLOW 6 and PESTPP-GLM. For the purposes of the tutorial(s), we have included executables in the tutorial repository. They are in the `bin_new` folder, organized by operating system and will be programmatically copied into the working dirs as needed. 
    # 
    # Some may prefer that executables be located in a folder that is cited in your computer's PATH environment variable. Doing so allows you to run them from a command prompt open to any other folder without having to include the full path to these executables in the command to run them. 
    # 
    # However, in situations where someone has several active projects and each may use difference versions of compiled binary codes, this may not be practical. In such cases, we can simply place the executables in the folder from which they will be executed.  So, let's copy the necessary executables into our working folder using a simple helper function:
    # 

    # %% [markdown]
    # Let's copy the original model folder into a new working directory, just to ensure we don't mess up the base files.

    # %%
    # folder containing original model files
    org_d = os.path.join('templates','make_gmdsi_freyberg','monthly_model_files_1lyr_newstress')

    # a dir to hold a copy of the org model files
    tmp_d = os.path.join('freyberg_mf6')

    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    shutil.copytree(org_d,tmp_d)

    get_bins(tmp_d)

    # get executables
    #hbd.prep_bins(tmp_d)
    # get dependency folders
    #hbd.prep_deps(tmp_d)

    # %% [markdown]
    # If you inspect the model folder, you will see that all the `MODFLOW6` model files have been written "externally". This is key for working with the `PstFrom` class (or with PEST(++) in general, really). Essentially, all pertinent model inputs have been written as independent files in either array or list format. This makes it easier for us to programmatically access and re-write the values in these files.
    # 
    # Array files contain a data type (usually floating points). List files will have a few columns that contain index information and then columns of floating point values (they have a tabular format; think `.csv` files or DataFrames). The `PstFrom` class provides methods for processing these file types into a PEST(++) dataset. 
    # 
    # 

    # %%
    os.listdir(tmp_d)

    # %% [markdown]
    # Now we need just a tiny bit of info about the spatial discretization of the model - this is needed to work out separation distances between parameters to build a geostatistical prior covariance matrix later.
    # 
    # Here we will load the flopy sim and model instance just to help us define some quantities later - flopy is ***not required*** to use the `PstFrom` class. ***Neither is MODFLOW***. However, at the time of writing, support for `SpatialReference` to spatially locate parameters is limited to structured grid models.
    # 
    # Load the simulation. Run it once to make sure it works and to ***make sure that model output files are in the folder***. 

    # %%
    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d)
    # load flow model
    gwf = sim.get_model()

    # run the model once to make sure it works
    pyemu.os_utils.run("mf6",cwd=tmp_d)
    # run modpath7
    pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim', cwd=tmp_d)

    # %% [markdown]
    # ### Spatial Reference
    # Now we can instantiate a `SpatialReference`. This will later be passed to `PstFrom` to assist with spatially locating parameters (e.g. pilot points and/or cell-by-cell parameters).  You can also use the flopy `modelgrid` class instance that is attached to the simulation, but `SpatialReference` is cleaner and faster for structured grids...

    # %%
    sr = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(tmp_d, "freyberg6.nam"),
            delr=gwf.dis.delr.array, delc=gwf.dis.delc.array)
    sr

    # %% [markdown]
    # ### Instantiate PstFrom
    # 
    # Now we can start to construct the PEST(++) interface by instantiating a `PstFrom` class instance. There are a few things that we need to specify up front:
    # 
    #  - the folder in which we currently have model files (e.g. `tmp_d`). PstFrom will copy all the files from this directory into a new "template" folder.
    #  - **template folder**: this is a folder in which the PEST dataset will be constructed - this folder will hold the model files plus all of the files needed to run PEST(++). This folder/dataset will form the template for subsequent deployment of PEST(++).
    #  - **longnames**: for backwards compatibility with PEST and PEST_HP (i.e. non-PEST++ versions), which have upper limits to parameter/observation names (PEST++ does not). Setting this value to False is only recommended if required. 
    #  - Whether the model is `zero based` or not.
    #  - (optional) the **spatial reference**, as previously discussed. This is only required if using `pyEMU` to define parameter spatial correlation. Alternatively, you can define these yourself or use utilities available in the PEST-suite. 

    # %%
    # specify a template directory (i.e. the PstFrom working folder)
    #template_ws = PST_TEMPLATE_DIR
    start_datetime="1-1-2008"
    # instantiate PstFrom
    pf = pyemu.utils.PstFrom(original_d=tmp_d, # where the model is stored
                                new_d=template_ws, # the PEST template folder
                                remove_existing=True, # ensures a clean start
                                longnames=True, # set False if using PEST/PEST_HP
                                spatial_reference=sr, #the spatial reference we generated earlier
                                zero_based=False, # does the MODEL use zero based indices? For example, MODFLOW does NOT
                                start_datetime=start_datetime, # required when specifying temporal correlation between parameters
                                echo=False) # to stop PstFrom from writing lots of information to the notebook; experiment by setting it as True to see the difference; useful for troubleshooting

    # %%
    os.listdir(template_ws)

    # %% [markdown]
    # So we see that when `PstFrom` is instantiated, it starts by copying the `original_d` to the `new_d`.  sweet as!

    # %% [markdown]
    # ### Observations
    # 
    # We now have a `PstFrom` instance assigned to the variable `pf`. For now it is only an empty container to which we can start adding "observations", "parameters" and other bits and bobs.
    # 
    # Let's start with observations because they are easier. `MODFLOW6` makes life even easier by recording observations in nicely organized .csv files. Isn't that a peach!
    # 
    # #### Freyberg Recap
    # As you may recall from the "*intro to Freyberg*" tutorial, the model is configured to record time series of head at observation wells, and flux at three locations along the river. These are recorded in external .csv files named `heads.csv` and `sfr.csv`, respectively. You should be able to see these files in the model folder.
    # 
    # Recall that each .csv houses records of observation time-series. Outputs are recorded for each simulated stress-period. The model starts with a single steady-state stress-period, followed by 24 monthly transient stress-periods. The steady-state and first 12 transient stress-periods simulate the history-matching period. The last 12 transient stress periods simulate future conditions (i.e. the prediction period).

    # %%
    # check the output csv file names
    for i in gwf.obs:
        print(i.output.obs_names)

    # %% [markdown]
    # Let's start with the 'heads.csv' file. First load it as a DataFrame to take a look:

    # %%
    df = pd.read_csv(os.path.join(template_ws,"heads.csv"),index_col=0)
    df.head()

    # %% [markdown]
    # As you can see, there are many columns, one for each observation site. Conveniently, * *cough* * they are named according to the cell layer, row and column. 
    # 
    # The values in the *.csv* file were generated by running the model. (***IMPORTANT!***) However, `PstFrom` assumes that values in this file are the *target* observation values, and they will be used to populate the PEST(++) dataset.  This lets the user quickly verify that the `PstFrom` process reproduces the same model output files - an important thing to test!
    # 
    # Now, you can and should change the observation values later on for the quantities that correspond to actual observation data.  This is the standard workflow when using `PstFrom` because it allows users to separate the PEST interface setup from the always-important process of setting observation values and weights. We address this part of the workflow in a separate tutorial.

    # %% [markdown]
    # #### Adding Observations
    # 
    # First, we will use the `PstFrom.add_observations()` method to add observations to our `pf` object. This method can use ***list-type*** files, where the data are organized in column/tabular format with one or more index columns and one or more data columns.  This method can also use ***array-type*** files, where the data are organized in a 2-D array structure (we will see this one later...)
    # 
    # We are going to tell `pf` which columns of this file contain observations. Values in these columns will be assigned to *observation values*.
    # 
    # We can also inform it if there is an index column (or columns). Values in this column will be included in the *observation names*. 
    # 
    # We could also specify which rows to include as observations. But observations are free...so why not keep them all! 
    # 
    # Let's add observations from `heads.csv`. The first column of this file records the time at which the value is simulated. Let's use that as the index column (this becomes useful later on to post-process results). We want all other columns as observation values.
    # 

    # %%
    hds_df = pf.add_observations("heads.csv", # the model output file to read
                                insfile="heads.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="hds") #prefix to all observation names; choose something logical and easy to find. We use it later on to select observations

    # %% [markdown]
    # Let's inspect what we just created. 
    # 
    # We can see that the `.add_observations()` method returned a dataframe with lots of useful info: 
    # 
    #  - the observation names that were formed (see `obsnme` column); note that these include lots of useful metadata like the column name, index value and so on;
    #  - the values that were read from `heads.csv` (see `obsval` column); 
    #  - some generic weights and group names; note that observations are grouped according to the column of the model output .csv. Alternatively, we could have specified a list of observation group names.

    # %%
    hds_df.head()

    # %% [markdown]
    # At this point, no PEST *control file* has been created, we have simply prepared to add these observations to the control file later. Everything is still only stored in memory. However, a PEST *instruction* file has been created in the template folder (`template_ws`):

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".ins")]

    # %% [markdown]
    # Blimey, wasn't that easy? Automatically monitoring thousands of model output quantities as observations into a PEST dataset becomes a breeze!
    # 
    # Let's quickly do the same thing for the SFR observations.

    # %%
    df = pd.read_csv(os.path.join(template_ws, "sfr.csv"), index_col=0)
    df.head()

    # %%
    # add the observations to pf
    sfr_df = pf.add_observations("sfr.csv", # the model output file to read
                                insfile="sfr.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="sfr") #prefix to all observation names

    # %% [markdown]
    # We also want to add observations of particle travel time and status. Unfortuanately the file written by MODPATH7 is not easily injestible by `PstFrom`. So we are going to need to "manually" construct an instruction file. We could add that now with the `PstFrom.add_observations_from_ins()` method, but we will wait and  add these after constructing the `Pst` object - soon!

    # %% [markdown]
    # ### Parameters
    # 
    # The `PstFrom.add_parameters()` method reads model input files and adds parameters to the PEST(++) dataset. Parameterization can be configured in several ways. 
    # 
    #  - model input files can be in array or list format;
    #  - parameters can be setup as different "types". Each value in model input files can (1) each be a separate parameter ("grid" scale parameters), (2) be grouped into "zones" or (3) all be treated as a single parameter ("constant" type). Alternatively, (4) parameters can be assigned to pilot points, from which individual parameter values are subsequently interpolated. `PstFrom` adds the relevant pre-processing steps to assign parameter values directly into the "model run" script.
    #  - parameter values can be setup as "direct", "multiplier" or "addend". This means the "parameter value" which PEST(++) sees can be (1) the same value the model sees, (2) a multiplier on the value in the existing/original model input file, or (3) a value which is added to the value in the existing/original model input file. This is very nifty and allows for some pretty advanced parameterization schemes by allowing mixtures of different types of parameters. `PstFrom` is designed to preferentially use parameters setup as multipliers (that is the default parameter type). This lets us preserve the existing model inputs and treat them as the mean of the prior parameter distribution. Once again, relevant pre-processing scripts are automatically added to the "model run" script (discussed later) so that the multiplicative and additive parameterization process is not something the user has to worry about.
    # 
    # 
    # #### Freyberg Recap
    # 
    # As discussed, all model inputs are stored in external files. Some are arrays. Others are lists. Recall that our model has 1 layer. It is transient. Hydraulic properties (Kh, Kv, Ss, Sy) vary in space. Recharge varies over both space and time. We have GHBs, SFR and WEL boundary conditions. GHB parameters are constant over time, but vary spatially. SFR inflow varies over time. Pumping rates of individual wells are uncertain in space and and time.
    # 
    # All of these have some degree of spatial and/or temporal correlation.
    # 
    # #### Geostatistical Structures
    # 
    # Parameter correlation plays a role in (1) regularization when giving preference to the emergence of patterns of spatial heterogeneity and (2) when specifying the prior parameter probability distribution (which is what regularization is enforcing!). Since we are all sophisticated and recognize the importance of expressing spatial and temporal uncertainty (e.g. heterogeneity) in the model inputs (and the corresponding spatial correlation in those uncertain inputs), let's use geostatistics to express uncertainty. To do that we need to define "geostatistical structures". 
    # 
    # For the sake of this tutorial, let's assume that heterogeneity for grid-scale parameters has a shorter correlation length than pilot point parameters.  So we will make two geostatistical structures for spatial parameters:

    # %%
    # exponential variogram for spatially varying parameters
    v_grid = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                        a=2000, # range of correlation; length units of the model. In our case 'meters'
                                        anisotropy=1.0, #name says it all
                                        bearing=0.0 #angle in degrees East of North corresponding to anisotropy ellipse
                                        )

    # geostatistical structure for spatially varying parameters
    grid_gs = pyemu.geostats.GeoStruct(variograms=v_grid, transform='log') 

    # plot the gs if you like:
    ax = grid_gs.plot()

    # %%
    # exponential variogram for spatially varying parameters
    v_pp = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                        a=30000, # range of correlation; length units of the model. In our case 'meters'
                                        anisotropy=1.0, #name says it all
                                        bearing=0.0 #angle in degrees East of North corresponding to anisotropy ellipse
                                        )

    # geostatistical structure for spatially varying parameters
    pp_gs = pyemu.geostats.GeoStruct(variograms=v_pp, transform='log') 

    # plot the gs if you like:
    _ = pp_gs.plot()

    # %%
    # exponential variogram for time varying parameters
    v_time = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                        a=180, # range of correlation; length time units (days)
                                        anisotropy=1.0, #do not change for 1-D time
                                        bearing=0.0 #do not change for 1-D time
                                        )

    # geostatistical structure for time varying parameters
    temporal_gs = pyemu.geostats.GeoStruct(variograms=v_time, transform='none') 

    # %% [markdown]
    # #### Add Parameters
    # 
    # Let's start by adding parameters of hydraulic properties that vary in space (but not time) and which are housed in array-type files (e.g. Kh, Kv, Ss, Sy). We will start by demonstrating step-by-step for Kh.
    # 
    # First, find all the external array files that contain Kh values. In our case, these are the files with "npf_k_" in the file name.

    # %%
    tag = "npf_k_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    print(files)

    # %% [markdown]
    # Let's setup multiple spatial scales of parameters for Kh. To do this we will use three of the parameter "types" described above. The coarse scale will be a `constant` single value for each array. The medium scale will `pilot points`. The finest scale will use parameters as the `grid` scale (a unique parameter for each model cell!)
    # 
    # Each scale of parameters will work with the others as multipliers with the existing Kh arrays. (This all happens at runtime as part of the "model run" script.) Think of the scales as dials that PEST(++) can turn to improve the fit. The "coarse" scale is one big dial that allows PEST to move everything at once - that is, change the mean of the entire Kh array. The "medium" dials are few (but not too many) that allow PEST to adjust broad areas, but not making everything move. The "fine" scales are lots of small dials that allow PEST(++) to have very detailed control, tweaking parameter values within very small areas. 
    # 
    # However, because we are working with parameter `multipliers`, we will need to specify two sets of parameter bounds: 
    #  - `upper_bound` and `lower_bound` are the standard control file bounds (the bounds on the parameters that PEST sees), while
    #  - `ult_ubound` and `ult_lbound` are bounds that are applied at runtime to the resulting (multiplied out) model input array that MODFLOW reads. 
    #  
    # Since we are using sets of multipliers, it is important to make sure we keep the resulting model input arrays within the range of realistic values.
    # 
    # #### Array Files
    # 
    # We will first demonstrate step-by-step for `freyberg6.npf_k_layer1.txt`. We will start with grid scale parameters. These are multipliers assigned to each individual value in the array.
    # 
    # We start by getting the idomain array. As our model has inactive cells, this helps us avoid adding unnecessary parameters. It is also required later when generating pilot points.

    # %%
    # get the IIDOMAIN array; in our case we only have one layer
    ib = gwf.dis.idomain.array[0]
    plt.imshow(ib)

    # %%
    f = 'freyberg6.npf_k_layer1.txt'

    # grid (fine) scale parameters
    df_gr = pf.add_parameters(f,
                    zone_array=ib, #as we have inactive model cells, we can avoid assigning these as parameters
                    par_type="grid", #specify the type, these will be unique parameters for each cell
                    geostruct=grid_gs, # the gestatisical structure for spatial correlation 
                    par_name_base=f.split('.')[1].replace("_","")+"gr", #specify a parameter name base that allows us to easily identify the filename and parameter type. "_gr" for "grid", and so forth.
                    pargp=f.split('.')[1].replace("_","")+"gr", #likewise for the parameter group name
                    lower_bound=0.2, upper_bound=5.0, #parameter lower and upper bound
                    ult_ubound=100, ult_lbound=0.01 # The ultimate bounds for multiplied model input values. Here we are stating that, after accounting for all multipliers, Kh cannot exceed these values. Very important with multipliers
                    )

    # %% [markdown]
    # As when adding observations,  `pf.add_parameters()` returns a dataframe. Take a look. You may recognize a lot of the information that appears in a PEST `*parameter data` section. All of this is still only housed in memory for now. We will write the PEST control file later on.
    # 
    # (HUGE) Note: The `add_parameters()` call above added grid-scale parameters, that is, one parameter per active model cell for HK.  While this is fine for this simple demo problem, in practice, adding grid-scale parameters (while sexy!) can be a massive headache re storage, memory, computation, pre- and post-processing, etc, to the point that is can derail the entire modeling analysis.  So unless you are have really, really (REALLY!!!) good prediction-driven reason to use grid-scale parameters, we suggest using (denser) pilot points as a sweet-spot between grid-scale and zones to express reasonable and relevant hetereogeneity.  

    # %%
    df_gr.head()

    # %% [markdown]
    # This `add_parameters()` call also wrote a template file that PEST(++) will use to populate the multiplier array at runtime:

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".tpl")]

    # %% [markdown]
    # Remember!  no PEST control file has been made yet. `PstFrom` is simply preparing to make a control file later...

    # %% [markdown]
    # Now, we add pilot point (medium scale) multiplier parameters to the same model input file. These multipliers are assigned to pilot points, which are subsequently interpolated to values in the array.
    # 
    # You can add pilot points in two ways:
    # 
    # 1. `PstFrom` can generate them for you on a regular grid or 
    # 2. you can supply `PstFrom` with existing pilot point location information in the form of a dataframe or a point-coverage shapefile. 
    # 
    # When you change `par_type` to "pilotpoints", by default, a regular grid of pilot points is set up using a default `pp_space` value of 10 (which is every 10th row and column). You can change this spacing by passing an integer to `pp_space` (as demonstrated below). 
    # 
    # Alternatively you can specify a filename or dataframe with pilot point locations. If you supply `pp_space` as a `str` it is assumed to be a filename. The extension is the guide: ".csv" for dataframe, ".shp" for shapefile (point-type). Anything else and the file is assumed to be a pilot points file type. The dataframe (or .csv file) must have "name", "x", and "y" as columns - it can have more, but must have those. 

    # %%
    # pilot point (medium) scale parameters
    df_pp = pf.add_parameters(f,
                        zone_array=ib,
                        par_type="pilotpoints",
                        geostruct=pp_gs,
                        par_name_base=f.split('.')[1].replace("_","")+"pp",
                        pargp=f.split('.')[1].replace("_","")+"pp",
                        lower_bound=0.2,upper_bound=5.0,
                        ult_ubound=100, ult_lbound=0.01,
                        pp_space=5) # `PstFrom` will generate a uniform grid of pilot points in every 4th row and column

    # %%
    fig,ax = plt.subplots(1,1,figsize=(4,6))
    ax.set_aspect("equal")
    ax.pcolormesh(sr.xcentergrid, sr.ycentergrid,ib)
    ax.scatter(df_pp.x,df_pp.y)

    # %% [markdown]
    # Lastly, add the constant (coarse) parameter multiplier. This is a single multiplier value applied to all values in the array. In practice, including a single constant parameter for each property can be an important parameter to include since it conceptually represents uncertainty in the mean property value

    # %%
    # constant (coarse) scale parameters
    df_cst = pf.add_parameters(f,
                        zone_array=ib,
                        par_type="constant",
                        geostruct=grid_gs,
                        par_name_base=f.split('.')[1].replace("_","")+"cn",
                        pargp=f.split('.')[1].replace("_","")+"cn",
                        lower_bound=0.2,upper_bound=5.0,
                        ult_ubound=100, ult_lbound=0.01)

    # %% [markdown]
    # Now we see three template files have been created:

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".tpl")]

    # %% [markdown]
    # Feel free to navigate to the `template_ws` and inspect these files.

    # %% [markdown]
    # We are also going to visualize the HK array later (since everyone is crazy about HK in groundwater modeling).  To make this easier, we can also add the HK array as "observations" in the pest control file.  This isn't always a good idea, especially if your model has heaps of rows/columns/nodes (in which case, you have lots of other problems to worry about!), but for reasonably sized models, this is an easy way to access to the HK array that the model sees, especially in an ensemble framework where there are lots of parameter sets (and therefore lots of HK arrays).  `PstFrom` makes this easy...

    # %%
    df = pf.add_observations(f,prefix="hk",zone_array=ib)
    df

    # %% [markdown]
    # See those nice observation names with the "i" and "j" values baked in?
    # 
    # Now, back to parameterization...We are going to be repeating this multiplier-parameter scheme for each parameter type, so let's write a function.

    # %%
    def add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100, add_coarse=True):
        if isinstance(f,str):
            base = f.split(".")[1].replace("_","")
        else:
            base = f[0].split(".")[1]
        # grid (fine) scale parameters
        pf.add_parameters(f,
                        zone_array=ib,
                        par_type="grid", #specify the type, these will be unique parameters for each cell
                        geostruct=grid_gs, # the gestatisical structure for spatial correlation 
                        par_name_base=base+"gr", #specify a parameter name base that allows us to easily identify the filename and parameter type. "_gr" for "grid", and so forth.
                        pargp=base+"gr", #likewise for the parameter group name
                        lower_bound=lb, upper_bound=ub, #parameter lower and upper bound
                        ult_ubound=uub, ult_lbound=ulb # The ultimate bounds for multiplied model input values. Here we are stating that, after accounting for all multipliers, Kh cannot exceed these values. Very important with multipliers
                        )
                        
        # pilot point (medium) scale parameters
        pf.add_parameters(f,
                            zone_array=ib,
                            par_type="pilotpoints",
                            geostruct=pp_gs,
                            par_name_base=base+"pp",
                            pargp=base+"pp",
                            lower_bound=lb, upper_bound=ub,
                            ult_ubound=uub, ult_lbound=ulb,
                            pp_space=5) # `PstFrom` will generate a uniform grid of pilot points in every 4th row and column
        if add_coarse==True:
            # constant (coarse) scale parameters
            pf.add_parameters(f,
                                zone_array=ib,
                                par_type="constant",
                                geostruct=grid_gs,
                                par_name_base=base+"cn",
                                pargp=base+"cn",
                                lower_bound=lb, upper_bound=ub,
                                ult_ubound=uub, ult_lbound=ulb)
        return

    # %% [markdown]
    # Let's speed through the other array parameter files.

    # %%

    # for Ss
    tag = "sto_ss"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    # only for layer 2 and 3; we aren't monsters
    for f in files[1:]: 
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=1e-7, uub=1e-3)

    # For Sy
    tag = "sto_sy"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    # only for layer 1
    f = files[0]
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)

    # For porosity (particle tracking...)
    tag = "ne_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files: 
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)


    # %%
    len([f for f in os.listdir(template_ws) if f.endswith(".tpl")])

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".tpl")]

    # %% [markdown]
    # Boom!  We just conquered property parameterization in a big way!

    # %% [markdown]
    # #### Spatial and Temporal Correlation
    # 
    # Now, you may be thinking "shouldn't recharge have temporal correlation as well?". 
    # 
    # Damn straight it should. Now, this requires a little trickery because native handling in spatiotemporal correlation is hard to do.  So what we are going to do is split this correlation into two setup of multiplier parameters.  One set of parameters will be constant in space but vary (and be correlated) in time.  The other set of multiplier parameters will be constant in time but vary (and be correlated) in space.  Since both of these sets of parameters are multipliers, we implicitly represent the concept that recharge is uncertain and correlated in both space and time.  Easy as!
    # 
    # First we need to construct a container of stress period datetimes. (This relies on specifying the start_datetime argument when instantiating `PstFrom`.) These datetime values will specify the position of parameters on the time-axis.
    # 
    # 

    # %%
    # build up a container of stress period start datetimes - this will
    # be used to specify the datetime of each multiplier parameter

    dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit='d')

    dts

    # %% [markdown]
    # If you use the same parameter group name (`pargp`) and same geostruct, `PstFrom` will treat parameters setup across different calls to `add_parameters()` as correlated - ***WARNING*** do not try to express spatial and temporal correlation together - as discussed above, #badtimes.  In this case, we want to express temporal correlation in the recharge multiplier parameters that are "constant" type in space so that there is one recharge multiplier parameter for each stress period that shares a parameter group name across different calls to `add_parameters`. So, we use the same parameter group names for each stress period data file, and specify the `datetime` and `geostruct` arguments.
    # 
    # Including temporal correlation introduces an additional challenge. Interpolation between points that share a common coordinate creates all types of trouble. We are going to have many parameters during each stress period (a single point on the time-axis). To get around this challenge we need to be a bit sneaky.
    # 
    # 
    # First, we will apply the multiple *spatial* scales of parameter multipliers (`constant`, `pilot point` and `grid`) as we did for hydraulic properties.  We do this for all recharge files at once, which tells `PstFrom` to broadcast (e.g. share) the same parameters for all of those files: 
    # 
    # 

    # %%
    # for Recharge; 
    tag = "rch_recharge"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s:f for s,f in zip(sp,files)}
    sp.sort()
    files = [d[s] for s in sp]
    print(files)
    # the spatial multiplier parameters; just use the same function
    add_mult_pars(files, lb=0.2, ub=5.0, ulb=0, uub=1e-3, add_coarse=False)

    # %%
    len([f for f in os.listdir(template_ws) if f.endswith(".tpl")])

    # %% [markdown]
    # Then, we will assign an additional `constant` multiplier parameter for each recharge stress-period file (so, a single multiplier for all recharge parameters for each stress period). We will specify temporal correlation for these `constant` multipliers. These will all have the same parameter group name, as discussed above. 

    # %%
    for f in files:   
        # multiplier that includes temporal correlation
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1  
        # add the constant parameters (with temporal correlation)
        pf.add_parameters(filenames=f,
                        zone_array=ib,
                        par_type="constant",
                        par_name_base=f.split('.')[1]+"tcn",
                        pargp=f.split('.')[1]+"tcn",
                        lower_bound=0.5, upper_bound=1.5,
                        ult_ubound=1e-3, ult_lbound=0,
                        datetime=dts[kper], # this places the parameter value on the "time axis"
                        geostruct=temporal_gs)

    # %% [markdown]
    # Sweet! Done!

    # %% [markdown]
    # ### List Files
    # 
    # Adding parameters from list-type files follows similar principles. As with observation files, they must be tabular. Certain columns are specified as index columns and are used to populate parameter names, as well as provide the parameters' spatial location. Other columns are specified as containing parameter values. 
    # 
    # Parameters can be `grid` or `constant`. As before, values can be assigned `directly`, as `multipliers` or as `additives`.
    # 
    # We will demonstrate for the boundary-condition input files. 
    # 
    # Starting off with GHBs. Let's inspect the folder. As you can see, there is a single input file (here we assume GHB parameters do not vary over time).

    # %%
    tag = "ghb_stress_period_data"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    print(files)

    # %% [markdown]
    # Since these boundaries are likely to be very influential, we want to include a robust representation of their uncertainty - both head and conductance and at multiple scales.  
    # 
    # Let's parameterize both GHB conductance and head:
    # 
    #  - For conductance, we shall use two scales of `multiplier` parameters (`constant` and `grid`).
    # 
    #  - For heads, multipliers are not ideal. Instead we will use `additive` parameters. Again, with a coarse and fine scale.
    # 
    #  **ATTENTION!** 
    #  
    #  Additive parameters by default get assigned an initial parameter value of zero. This can be problematic later on when computing the derivatives. Be sure to either apply a parameter offset, or use "absolute" increment types in the parameter group section (we will implement the latter option further on in the current tutorial.)

    # %%
    tag = "ghb_stress_period_data"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]

    for f in files:
        # constant and grid scale multiplier conductance parameters
        name = 'ghbcond'
        pf.add_parameters(f,
                            par_type="grid",
                            geostruct=grid_gs,
                            par_name_base=name+"gr",
                            pargp=name+"gr",
                            index_cols=[0,1,2], #column containing lay,row,col
                            use_cols=[4], #column containing conductance values
                            lower_bound=0.1,upper_bound=10.0)
        pf.add_parameters(f,
                            par_type="constant",
                            geostruct=grid_gs,
                            par_name_base=name+"cn",
                            pargp=name+"cn",
                            index_cols=[0,1,2],
                            use_cols=[4],  
                            lower_bound=0.1,upper_bound=10.0,
                            ult_lbound=0.01, ult_ubound=100) #absolute limits

        # constant and grid scale additive head parameters
        name = 'ghbhead'
        pf.add_parameters(f,
                            par_type="grid",
                            geostruct=grid_gs,
                            par_name_base=name+"gr",
                            pargp=name+"gr",
                            index_cols=[0,1,2],
                            use_cols=[3],   # column containing head values
                            lower_bound=-2.0,upper_bound=2.0,
                            par_style="a", # specify additive parameter
                            transform="none", # specify not log-transform
                            ult_lbound=32.5, ult_ubound=42) #absolute limits; make sure head is never lower than the bottom of layer1
        pf.add_parameters(f,
                            par_type="constant",
                            geostruct=grid_gs,
                            par_name_base=name+"cn",
                            pargp=name+"cn",
                            index_cols=[0,1,2],
                            use_cols=[3],
                            lower_bound=-2.0,upper_bound=2.0, 
                            par_style="a", 
                            transform="none",
                            ult_lbound=32.5, ult_ubound=42) 

    # %% [markdown]
    # Easy peasy.
    # 
    # Now, this will make some people uncomfortable, but how well do we really ever know historic water use flux rates in space and in time? hmmm, not really! And just a little uncertainty in historic water use can result in large changes in simulated water levels...So let's add parameters to represent that uncertainty in the model inputs.
    # 
    # For wells it may not (or it may...) make sense to include spatial correlation. Here we will assume temporal correlation - its reasonable that pumping rates today will be similar to pumping rates yesterday. 
    # 
    # Pumping rates for different stress periods are in separate files. We will call `.add_parameters()` for each file. But we want to specify correlation between parameters in different files. As explained above for recharge, we do this with the parameter group name.
    # 
    # OK, let's get started.
    # 

    # %% [markdown]
    # As discussed above, including temporal correlation introduces an additional challenge. We use the same approach described for recharge parameters:
    # 
    #  - First, we will assign a `constant` multiplier parameter for each WEL stress-period file (so, a single multiplier for all well pumping rates for each stress period). We will specify temporal correlation for these `constant` multipliers.
    # 
    #  - Then, we will also have `grid` type multiplier parameters for each WEL stress period file (so, multipliers for individual well pumping rate during each stress period). These will not include (temporal) correlation. (We could in principle include spatial correlation here if we wanted to; but let's not).

    # %%
    files = [f for f in os.listdir(template_ws) if "wel_stress_period_data" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s:f for s,f in zip(sp,files)}
    sp.sort()
    files = [d[s] for s in sp]

    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1  
        
        # add the constant parameters (with temporal correlation)
        pf.add_parameters(filenames=f,
                            index_cols=[0,1,2], #columns that specify cell location
                            use_cols=[3],       #columns with parameter values
                            par_type="constant",    #each well will be adjustable
                            par_name_base="welcst",
                            pargp="welcst", 
                            upper_bound = 4, lower_bound=0.25,
                            datetime=dts[kper], # this places the parameter value on the "time axis"
                            geostruct=temporal_gs)
        
        # add the grid parameters; each individual well
        pf.add_parameters(filenames=f,
                            index_cols=[0,1,2], #columns that specify cell location 
                            use_cols=[3],       #columns with parameter values
                            par_type="grid",    #each well will be adjustable
                            par_name_base="welgrd",
                            pargp="welgrd", 
                            upper_bound = 4, lower_bound=0.25,
                            datetime=dts[kper]) # this places the parameter value on the "time axis"
                        

    # %% [markdown]
    # And finally, our favorite (not!) boundary-condition: SFR.
    # 
    # Let's parameterize conductance (time-invariant) and inflow (time-variant).

    # %%
    # SFR conductance
    tag = "sfr_packagedata"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    assert len(files) == 1 # There can be only one! It is tradition. Jokes.
    print(files)

    f = files[0]
    # constant and grid scale multiplier conductance parameters
    name = "sfrcond"
    pf.add_parameters(f,
                    par_type="grid",
                    geostruct=grid_gs,
                    par_name_base=name+"gr",
                    pargp=name+"gr",
                    index_cols=[0,2,3],
                    use_cols=[9],
                    lower_bound=0.1,upper_bound=10.0)
    pf.add_parameters(f,
                    par_type="constant",
                    geostruct=grid_gs,
                    par_name_base=name+"cn",
                    pargp=name+"cn",
                    index_cols=[0,2,3],
                    use_cols=[9],
                    lower_bound=0.1,upper_bound=10.0,
                    ult_lbound=0.001, ult_ubound=100) #absolute limits

    # %%
    # SFR inflow
    files = [f for f in os.listdir(template_ws) if "sfr_perioddata" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s:f for s,f in zip(sp,files)}
    sp.sort()
    files = [d[s] for s in sp]
    print(files)
    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1  
        # add the parameters
        pf.add_parameters(filenames=f,
                            index_cols=[0], #reach number
                            use_cols=[2],   #columns with parameter values
                            par_type="grid",    
                            par_name_base="sfrgr",
                            pargp="sfrgr", 
                            upper_bound = 10, lower_bound=0.1, #don't need ult_bounds because it is a single multiplier
                            datetime=dts[kper], # this places the parameter value on the "time axis"
                            geostruct=temporal_gs)

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".tpl")]

    # %% [markdown]
    # Damn!  we just parameterized many recognized sources of model input uncertainty at several spatial and temporal scales.  And we expressed spatial and temporal correlation in those parameters.  One last set of parameters that we will need later for sequential data assimilation - initial conditions:

    # %%
    files = [f for f in os.listdir(template_ws) if "ic_strt" in f and f.endswith(".txt")]
    files

    # %%
    for f in files:
        base = f.split(".")[1].replace("_","")
        df = pf.add_parameters(f,par_type="grid",par_style="d",
                        pargp=base,par_name_base=base,upper_bound=50,
                        lower_bound=15,zone_array=ib,transform="none")
        print(df.shape)

    # %% [markdown]
    # ### The Forward Run Script
    # 
    # OK! So, we almost have all the base building blocks for a PEST(++) dataset. We have some (1) observations and some (2) parameters. We are still missing (3) the "forward run" script. Recall that in the PEST world, the "model" is not just the numerical model (e.g. MODFLOW). Instead it is a composite of the numerical model (or models) and pre- and post-processing steps, encapsulated in a "forward run" script which can be called from the command line. This command line instruction is what PEST(++) sees as "the model". During execution, PEST(++) writes values to parameter files, runs "the model", and then reads values from the observation files.
    # 
    # `PstFrom` automates the generation of such a script when constructing the PEST control file. The script is written to file named `forward_run.py`. It is written in Python (this is not a PEST(++) requirement, merely a convenience...we are working in Python after all...). 
    # 
    # How about we see that in action? Magic time! Let's create the PEST control file.

    # %%
    pst = pf.build_pst()

    # %% [markdown]
    # Boom! Done. (Well almost.) Check the folder. You should see a new .pst file and the `forward_run.py` file. By default, the .pst file is named after the original model folder name. 

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".py") or f.endswith(".pst") ]

    # %% [markdown]
    # We will get to the `pst` object later on (see also the "intro to pyemu" tutorial notebook). For now, let's focus on the `forward_run.py` script. It is printed out below.
    # 
    # This script does a bunch of things:
    #  - it loads necessary dependencies
    #  - it removes model output files to avoid the possibility of files from a previous model run being read by mistake;
    #  - it runs pre-processing steps (see `pyemu.helpers.apply_list_and_array_pars()`;
    #  - it executes system commands (usually running the simulator, i.e. MODFLOW). (*This is still missing. We will demonstrate next.*)
    #  - it executes post-processing steps; (*for now there aren't any*)
    #  - ...it washes the dishes (sorry, no it doesn't...this feature is still in development).

    # %%
    _ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]

    # %% [markdown]
    # That's pretty amazing. But as we just saw, we still need to add commands to actually run the model.
    # 
    # `PstFrom` allows you to pass a list of system commands which will be executed in sequence. It also has methods for including Python functions that run before or after the system commands. These make pre-/post-processing a piece of cake. In fact, we have already started to add to it. Remember all of the multiplier and additive parameters we setup? These all require pre-processing steps to convert the PEST-generated multipliers into model input values. `PstFrom` will automatically add these functions to the `forward_run.py` script. Nifty, hey?
    # 
    # Next we will demonstrate how to specify the system commands and add Python functions as processing steps.
    # 
    # #### Sys Commands
    # 
    # Let's start by adding a command line instruction. These are stored as a list in `PstFrom.mod_sys_cmds`, which is currently empty. 

    # %%
    pf.mod_sys_cmds 

    # %% [markdown]
    # To run a MODFLOW6 model from the command line, you can simply execute `mf6` in the model folder. So, we can add this command by appending it to the list. (Do this only once! Every time you append 'mf6' results in an additional call to MODFLOW6, meaning the model would be run multiple times.)
    # 
    # `PstFrom` will add a line to `forward_run.py` w

    # %%
    pf.mod_sys_cmds.append("mf6") #do this only once
    pf.mod_sys_cmds

    # %% [markdown]
    # We also need to run MODPATH7, so we need to add that to the list of system commands. In this case we also need to specify the modpath sim file:

    # %%
    pf.mod_sys_cmds.append("mp7 freyberg_mp.mpsim") #do this only once
    pf.mod_sys_cmds

    # %% [markdown]
    # OK, now let's re-build the Pst control file and check out the changes to the `forward_run.py` script.
    # 
    # You should see that `pyemu.os_utils.run(r'mf6')` has been added after the pre-processing functions.

    # %%
    pst = pf.build_pst()

    _ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]

    # %% [markdown]
    # #### Extra pre- and post-processing functions
    # 
    # You will also certainly need to include some additional processing steps.  These are supported through the `PstFrom.pre_py_cmds` and `PstFrom.post_py_cmds`, which are lists for pre and post model run python commands and `PstFrom.pre_sys_cmds` and `PstFrom.post_sys_cmds`, which are lists for pre and post model run system commands (these are wrapped in `pyemu.os_utils.run()`.  
    # 
    # But what if your additional steps are actually an entire python function? Well, we got that too! `PstFrom.add_py_function()`. This method allows you to get functions from another (pre-prepared) python source file and add them to the `forward_run.py` script. We will demonstrate this to post-process model observations after each run.

    # %% [markdown]
    # Now let's see this py-sauce in action: we are going to add a little post-processing function to extract the final simulated water level for all model cells for the last stress period from the MF6 binary headsave file and save them to ASCII format so that PEST(++) can read them with instruction files.  And, while we are at it, let's also extract the global water budget info from the MF6 listing file and store it in dataframes - these are usually good numbers to watch!  We will need the simulated water level arrays later for sequential data assimilation (wouldn't it be nice if MF6 supported the writing of ASCII format head arrays?).  Anyway, this function is stored in the "helpers.py" script which you can find in the tutorial folder.

    # %%
    pf.add_py_function("run_pstfrom.py","extract_hds_arrays_and_list_dfs()",is_pre_cmd=False)

    # %% [markdown]
    # That last argument - `is_pre_cmd` tells `PstFrom` if the python function should be treated as a pre-processor or a post-processor. So we have added that post-processor, but we still need to setup pest observations for those ASCII head arrays.  Let's do that by first calling that function to operate once within the `template_ws` to generate the arrays and then we can add them with `add_observations()`:  

    # %%

    test_extract_hds_arrays(template_ws)

    # %%
    files = [f for f in os.listdir(template_ws) if f.startswith("hdslay")]
    files

    # %%
    for f in files:
        pf.add_observations(f,prefix=f.split(".")[0],obsgp=f.split(".")[0])

    # %%
    for f in ["inc.csv","cum.csv"]:
        df = pd.read_csv(os.path.join(template_ws,f),index_col=0)
        pf.add_observations(f,index_cols=["totim"],use_cols=list(df.columns.values),
                            prefix=f.split('.')[0],obsgp=f.split(".")[0])

    # %% [markdown]
    # Crushed it!

    # %% [markdown]
    # #### Secondary Observations
    # 
    # Often it is useful to include "secondary model outcomes" as observations. These can be important components in a history-matching dataset to tease out specific aspects of system behaviour (e.g. head differences between aquifer layers to inform vertical permeabilities). Or they may be simple summaries of modelled outputs which are of interest for a prediction (e.g. minimum simulated head over a given period).
    # 
    # If you inspect the tutorial folder you will find a file named `helpers.py`. This is a python source file which we have prepared for you. (Open it to see how it is organized.) This is not standard `pyemu` functionality, but we provide it here as an example of incorporating custom code for observation processing. The sky's the limit! It contains a function named `process_secondary_obs()`. This function reads the model output .csv files, processes them and writes a series of new observation .csv files. These new files contain the temporal-differences between head and SFR observations. The new .csv files are named `heads.tdiff.csv`and `sfr.tdiff.csv`, respectively.
    # 
    # First, let's load the function here and run it so you can see what happens. (And to make sure that the observation files are in the template folder!) 
    # 
    # Run the next cell, then inspect the template folder. You should see several new csv files. These are the new secondary observations calculated by the post-processing function.

    # %%
    # run the helper function
    process_secondary_obs(ws=template_ws)

    # %%
    [f for f in os.listdir(template_ws) if f.endswith(".csv")]

    # %% [markdown]
    # OK, so now let's add this function to the `forward_run.py` script.

    # %%
    pf.add_py_function("run_pstfrom.py", # the file which contains the function
                        "process_secondary_obs(ws='.')", #the function, making sure to specify any arguments it may require
                        is_pre_cmd=False) # whether it runs before the model system command, or after. In this case, after.

    # %% [markdown]
    # And, boom! Bob's your uncle. As easy as that.
    # 
    # Now, of course we want to add these observations to `PstFrom` as well:

    # %%

    df = pd.read_csv(os.path.join(template_ws, "sfr.tdiff.csv"), index_col=0)
    _ = pf.add_observations("sfr.tdiff.csv", # the model output file to read
                                insfile="sfr.tdiff.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="sfrtd") #prefix to all observation 
                                
    df = pd.read_csv(os.path.join(template_ws, "heads.tdiff.csv"), index_col=0)
    _ = pf.add_observations("heads.tdiff.csv", # the model output file to read
                                insfile="heads.tdiff.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="hdstd") #prefix to all observation names

    # %% [markdown]
    # Remember to re-build the Pst control file:

    # %%
    pst = pf.build_pst()

    # %%
    _ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]

    # %% [markdown]
    # Now we see that `extract_hds_array_and_list_dfs()` has been added to the forward run script and it is being called after MF6 runs. 

    # %%
    obs = pst.observation_data
    obs

    # %% [markdown]
    # ### After Building the Control File
    # 
    # At this point, we can do some additional modifications that would typically be done that are problem specific.  Here we can tweak the setup, specifying things such as observation weights, parameter bounds, transforms, control data, etc.
    # 
    # Note that any modifications made after calling `PstFrom.build_pst()` will only exist in memory - you need to call `pf.pst.write()` to record these changes to the control file on disk.  Also note that if you call `PstFrom.build_pst()` after making some changes, these changes will be lost.  
    # 
    # For the current case, the main thing we haven't addressed are the observations from custom *.ins files,  observation weights, parameter group INCTYP's and forecasts.
    # 
    # We will do so now.

    # %% [markdown]
    # #### Add Observations from INS files
    # 
    # Recall that we wish to include observations of particle end time and status. As mentioned earlier, MP7 output files are not ina nicely organized tabular format - so we need to construct a custom instruction file. We will do this now:

    # %%
    # write a really simple instruction file to read the MODPATH end point file
    out_file = "freyberg_mp.mpend"
    ins_file = out_file + ".ins"
    with open(os.path.join(template_ws, ins_file),'w') as f:
        f.write("pif ~\n")
        f.write("l7 w w w w !part_status! w w !part_time!\n")

    # %% [markdown]
    # Now add these observations to the `Pst`:

    # %%
    pst.add_observations(ins_file=os.path.join(template_ws, ins_file),
                        out_file=os.path.join(template_ws, out_file),
                                pst_path='.')

    # and then check what changed                            
    obs = pst.observation_data
    obs.loc[obs.obsnme=='part_status', 'obgnme'] = 'part'
    obs.loc[obs.obsnme=='part_time', 'obgnme'] = 'part'

    obs.iloc[-2:]

    # %% [markdown]
    # #### Parameters with Zero as Initial Value
    # 
    # Recall that we assigned additive parameters to the GHB heads. Our initial parameter values for these parameter types were set as 0 (zero). This creates a wee bit of trouble when calculating derivatives - this is only an issue for algorithms that use finite-difference derivatives. There are a couple of ways we could get around it. One way is to add an "offset" to the parameter initial values and to the parameter bounds. Another is to use "absolute" increment types (INCTYP). See the PEST manual or PEST++ user guides for descriptions of increment types. 
    # 
    # We will apply both here. 
    # 
    # We will assign INCTYP as 'absolute'. We will leave DERINC as 0.01 (the default). It is a reasonable value in this case.

    # %%
    head_pargps = [i for i in pst.adj_par_groups if 'head' in i]
    head_pargps

    # %%
    pst.parameter_groups.loc[head_pargps, 'inctyp'] = 'absolute'

    # %% [markdown]
    # Now add the "offset" to parameter data entries:

    # %%
    par = pst.parameter_data
    par_names = par.loc[par.parval1==0].parnme

    par.loc[par_names].head()

    # %%
    offset = -10
    par.loc[par_names, 'offset'] = offset
    par.loc[par_names, ['parval1', 'parlbnd', 'parubnd']] -= offset

    par.loc[par_names].head()

    # %% [markdown]
    # #### Forecasts
    # 
    # For most models there is a forecast/prediction that someone needs. Rather than waiting until the end of the project, the forecast should be entered into your thinking and workflow __right at the beginning__.  Here we do this explicitly by monitoring the forecasts as "observations" in the control file.  This way, for every PEST(++) analysis we do, we can watch what is happening to the forecasts - #winning
    # 
    # The optional PEST++ `++forecasts` control variable allows us to provide the names of one or more observations featured in the "observation data" section of the PEST control file; these are treated as predictions in FOSM predictive uncertainty analysis by PESTPP-GLM. It is also a convenient way to keep track of "forecast" observations (makes post-processing a wee bit easier later on).
    # 
    # Recall that, for our synthetic case we are interested in forecasting:
    # 
    #  - groundwater level in the upper layer at row 9 and column 1 (site named "trgw-0-9-1") in stress period 22 (time=640);
    #  - the "tailwater" surface-water/groundwater exchange during stress period 13 (time=367); and
    #  - the "headwater" surface-water/groundwater exchange at stress period 22 (time=640).
    #  - the particle travel time.
    # 

    # %%
    forecasts =[
                'oname:sfr_otype:lst_usecol:tailwater_time:4383.5',
                'oname:sfr_otype:lst_usecol:headwater_time:4383.5',
                'oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5',
                'part_time'
                ]

    forecasts

    # %%
    fobs = obs.loc[forecasts,:]
    fobs

    # %% [markdown]
    # We will just set this optional pest++ argument because it will trigger certain automatic behavior later in PESTPP-GLM

    # %%
    pst.pestpp_options['forecasts'] = forecasts

    # %% [markdown]
    # And a couple of run mgmt options just to make sure we dont fall victim to an infinite loops that seem to happen with modpath7 every once and a while:

    # %%
    pst.pestpp_options["overdue_giveup_fac"] = 3
    pst.pestpp_options["overdue_giveup_minutes"] = 0.5

    # %% [markdown]
    # ### Re-write the Control File!

    # %% [markdown]
    # Make sure to re-**write** the PEST control file. But beware, if you re-**build** the `Pst`, all these changes will be lost.

    # %%
    pst.write(os.path.join(template_ws, 'pest.pst'),version=2)

    # %% [markdown]
    # So that was pretty epic. We now have a (very) high-dimensional PEST interface that includes secondary observations, as well as forecasts, ready to roll. 
    # 
    # If you inspect the folder, you will see PEST control file and all the necessary instruction and template files. Because we have >10k parameters, version 2 of the PEST control file was written by default. 
    # 
    # Shall we check that it works? Let's run PEST once (i.e. with NOPTMAX=0). Now, by default, noptmax is set to zero. But just to check:

    # %%
    pst.control_data.noptmax

    # %% [markdown]
    # OK, so when we run PEST it will call the model once and then stop. If the next cell is successful, then everything is working. Check the folder, you should see PEST output files. (We will go into these and how to process PEST outcomes in subsequent tutorials).

    # %%
    pyemu.os_utils.run('pestpp-ies pest.pst', cwd=template_ws)

    # %% [markdown]
    # Recall that we assigned observation values generated from the "base model run"? If we setup everything correctly, this means that PEST should have obtained residuals very close to zero. As mentioned, this is a good way to check for problems early on.
    # 
    # Let's check the Phi recorded in the *.iobj file (could also check the *.rec or *.rei files).

    # %% [markdown]
    #  Sweet! Zero. All is well.

    # %% [markdown]
    # ### Prior Parameter Covariance Matrix
    # 
    # One the major reasons `PstFrom` was built is to help with building the Prior - both covariance matrix and ensemble - with geostatistical correlation.  Remember all that business above related to geostatistical structures and correlations?  This is where it pays off.
    # 
    # Let's see how this works.  For cases with less than about 30,000 parameters, we can actually generate and visualize the prior parameter covariance matrix.  If you have more parameters, this matrix may not fit in memory.  But, not to worry, `PstFrom` has some trickery to help generate the geostatistical prior ensemble even in cases where the number of parameters is greater than 30,000. 
    # 

    # %%
    # build the prior covariance matrix and store it as a compressed binary file (otherwise it can get huge!)
    # depending on your machine, this may take a while...
    if pf.pst.npar < 35000:  #if you have more than about 35K pars, the cov matrix becomes hard to handle
        cov = pf.build_prior(fmt='coo', filename=os.path.join(template_ws,"prior_cov.jcb"))
        # and take a peek at a slice of the matrix
        try: 
            x = cov.x.copy()
            x[x==0] = np.NaN
            plt.imshow(x[:1000,:1000])
        except:
            pass

    # %% [markdown]
    # snap!  That big block must be a grid-scale parameter group...

    # %%
    cov.row_names[:10]

    # %% [markdown]
    # And now generate a prior parameter ensemble. This step is relevant for using pestpp-ies in subsequent tutorials. Note: you do not have to call `build_prior()` before calling `draw()`!

    # %%
    pe = pf.draw(num_reals=1000, use_specsim=True) # draw parameters from the prior distribution
    pe.enforce() # enforces parameter bounds
    pe.to_binary(os.path.join(template_ws,"prior_pe.jcb")) #writes the parameter ensemble to binary file
    assert pe.shape[1] == pst.npar

    pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'
    pst.write(os.path.join(template_ws, 'pest.pst'),version=2)



    return


def set_weights(t_d):

    # %%
    pst_file = "pest.pst"
    pst = pyemu.Pst(os.path.join(t_d, pst_file))

    # %% [markdown]
    # When we constructed the PEST dataset (in the "pstfrom pest setup" tutorial) we simply identified what model outputs we wanted PEST to "observe". In doing so, `pyemu.PstFrom` assigned observation target values that it found in the existing model output files. (Which conveniently allowed us to test whether out PEST setup was working correctly). All observation weights were assigned a default value of 1.0. 
    # 
    # As a reminder:

    # %%
    obs = pst.observation_data
    obs.head()

    # %% [markdown]
    # As mentioned above, we need to do several things:
    #  - replace observation target values (the `obsval` column) with corresponding values from "measured data";
    #  - assign meaningful weights to history matching target observations (the `weight` column);
    #  - assign zero weight to observations that should not affect history matching.

    # %% [markdown]
    # Let's start off with the basics. First set all weights to zero. We will then go through and assign meaningful weights only to relevant target observations. 

    # %%
    #check for nonzero weights
    obs.weight.value_counts()

    # %%
    # assign all weight zero
    obs.loc[:, 'weight'] = 0

    # check for non zero weights
    obs.weight.unique()

    # %% [markdown]
    # ### Measured Data
    # 
    # In most data assimilation contexts you will have some relevant measured data (e.g. water levels, river flow rates, etc.) which correspond to simulated model outputs. These will probably not coincide exactly with your model outputs. Are the wells at the same coordinate as the center of the model cell? Do measurement times line up nicely with model output times? Doubt it. And if they do, are single measurements that match model output times biased? And so on... 
    # 
    # A modeller needs to ensure that the observation values assigned in the PEST control file are aligned with simulated model outputs. This will usually require some case-specific pre-processing. Here we are going to demonstrate __an example__ - but remember, every case is different!

    # %% [markdown]
    # First, let's access our dataset of "measured" observations.

    # %%
    obs_csv = os.path.join('templates','make_gmdsi_freyberg',"obs_data.csv")
    assert os.path.exists(obs_csv)
    obs_data = pd.read_csv(obs_csv)
    obs_data.set_index('site', inplace=True)
    obs_data.iloc[:5]

    # %% [markdown]
    # As you can see, we have measured data at daily intervals. But our model simulates monthly stress periods. So what observation value do we use? 
    # 
    # One option is to simply sample measured values from the data closest to our simulated output. The next cell does this, with a few checks along the way:

    # %%
    #just pick the nearest to the sp end
    model_times = pst.observation_data.time.dropna().astype(float).unique()
    # get the list of osb names for which we have data
    obs_sites =  obs_data.index.unique().tolist()

    # restructure the observation data 
    es_obs_data = []
    for site in obs_sites:
        site_obs_data = obs_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        elif isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            site_obs_data = site_obs_data.reindex(model_times,method="nearest")

        if site_obs_data.shape != site_obs_data.dropna().shape:
            print("broke",site)
        es_obs_data.append(site_obs_data)
    es_obs_data = pd.concat(es_obs_data,axis=0,ignore_index=True)
    es_obs_data.shape

    # %% [markdown]
    # Right then...let's plot our down-sampled measurement data and compare it to the original high-frequency time series.
    # 
    # The next cell generates plots for each time series of measured data. Blue lines are the original high-frequency data. The marked red line is the down-sampled data. What do you think? Does sampling to the "closest date" capture the behaviour of the time series? Doesn't look too good...It does not seem to capture the general trend very well.
    # 
    # Let's try something else instead.

    # %%
    for site in obs_sites:
        #print(site)
        site_obs_data = obs_data.loc[site,:]
        es_site_obs_data = es_obs_data.loc[es_obs_data.site==site,:].copy()
        es_site_obs_data.sort_values(by="time",inplace=True)
        #print(site,site_obs_data.shape)
        fig,ax = plt.subplots(1,1,figsize=(10,2))
        ax.plot(site_obs_data.time,site_obs_data.value,"b-",lw=0.5)
        #ax.plot(es_site_obs_data.datetime,es_site_obs_data.value,'r-',lw=2)
        ax.plot(es_site_obs_data.time,es_site_obs_data.value,'r-',lw=1,marker='.',ms=10)
        ax.set_title(site)
    #plt.show()

    # %% [markdown]
    # This time, let's try using a moving-average instead. Effectively this is applying a low-pass filter to the time-series, smoothing out some of the spiky noise. 
    # 
    # The next cell re-samples the data and then plots it. Measured data sampled using a low-pass filter is shown by the marked green line. What do you think? Better? It certainly does a better job at capturing the trends in the original data! Let's go with that.

    # %%
    ess_obs_data = {}
    for site in obs_sites:
        #print(site)
        site_obs_data = obs_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times,method="nearest")
        #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
        ess_obs_data[site] = sm_site_obs_data
        
        es_site_obs_data = es_obs_data.loc[es_obs_data.site==site,:].copy()
        es_site_obs_data.sort_values(by="time",inplace=True)
        fig,ax = plt.subplots(1,1,figsize=(10,2))
        ax.plot(site_obs_data.time,site_obs_data.value,"b-",lw=0.25)
        ax.plot(es_site_obs_data.time,es_site_obs_data.value,'r-',lw=1,marker='.',ms=10)
        ax.plot(sm_site_obs_data.index,sm_site_obs_data.values,'g-',lw=0.5,marker='.',ms=10)
        ax.set_title(site)
    #plt.show()
    ess_obs_data = pd.DataFrame(ess_obs_data)
    ess_obs_data.shape

    # %% [markdown]
    # ### Update Target Observation Values in the Control File
    # 
    # Right then - so, these are our smoothed-sampled observation values:

    # %%
    ess_obs_data.head()

    # %% [markdown]
    # Now we are confronted with the task of getting these _processed_ measured observation values into the `Pst` control file. Once again, how you do this will end up being somewhat case-specific and will depend on how your observation names were constructed. For example, in our case we can use the following function (we made it a function because we are going to repeat it a few times):

    # %%
    def update_pst_obsvals(obs_names, obs_data):
        """obs_names: list of selected obs names
        obs_data: dataframe with obs values to use in pst"""
        # for checking
        org_nnzobs = pst.nnz_obs
        # get list of times for obs name suffixes
        time_str = obs_data.index.map(lambda x: f"time:{x}").values
        # empyt list to keep track of missing observation names
        missing=[]
        for col in obs_data.columns:
            # get obs list suffix for each column of data
            obs_sufix = col.lower()+"_"+time_str
            for string, oval, time in zip(obs_sufix,obs_data.loc[:,col].values, obs_data.index.values):
                    
                    if not any(string in obsnme for obsnme in obs_names):
                        if string.startswith("trgw-2"):
                            pass
                        else:
                            missing.append(string)
                    # if not, then update the pst.observation_data
                    else:
                        # get a list of obsnames
                        obsnme = [ks for ks in obs_names if string in ks] 
                        assert len(obsnme) == 1,string
                        obsnme = obsnme[0]
                        # assign the obsvals
                        obs.loc[obsnme,"obsval"] = oval
                        # assign a generic weight
                        if time > 3652.5 and time <=4018.5:
                            obs.loc[obsnme,"weight"] = 1.0
        # checks
        #if (pst.nnz_obs-org_nnzobs)!=0:
        #    assert (pst.nnz_obs-org_nnzobs)==obs_data.count().sum()
        if len(missing)==0:
            print('All good.')
            print('Number of new nonzero obs:' ,pst.nnz_obs-org_nnzobs) 
            print('Number of nonzero obs:' ,pst.nnz_obs)  
        else:
            raise ValueError('The following obs are missing:\n',missing)
        return

    # %%
    pst.nnz_obs_groups

    # %%
    # subselection of observation names; this is because several groups share the same obs name suffix
    obs_names = obs.loc[obs.oname.isin(['hds', 'sfr']), 'obsnme']

    # run the function
    update_pst_obsvals(obs_names, ess_obs_data)

    # %%
    pst.nnz_obs_groups

    # %%
    pst.observation_data.oname.unique()

    # %% [markdown]
    # So that has sorted out the absolute observation groups. But remember the 'sfrtd' and 'hdstd' observation groups? Yeah that's right, we also added in a bunch of other "secondary observations" (the time difference between observations) as well as postprocessing functions to get them from model outputs. We need to get target values for these observations into our control file as well!
    # 
    # Let's start by calculating the secondary values from the absolute measured values. In our case, the easiest is to populate the model output files with measured values and then call our postprocessing function.
    # 
    # Let's first read in the SFR model output file, just so we can see what is happening:

    # %%
    obs_sfr = pd.read_csv(os.path.join(t_d,"sfr.csv"),
                        index_col=0)

    obs_sfr.head()

    # %% [markdown]
    # Now update the model output csv files with the smooth-sampled measured values:

    # %%
    def update_obs_csv(obs_csv):
        obsdf = pd.read_csv(obs_csv, index_col=0)
        check = obsdf.copy()
        # update values in reelvant cols
        for col in ess_obs_data.columns:
            if col in obsdf.columns:
                obsdf.loc[:,col] = ess_obs_data.loc[:,col]
        # keep only measured data columns; helps for vdiff and tdiff obs later on
        #obsdf = obsdf.loc[:,[col for col in ess_obs_data.columns if col in obsdf.columns]]
        # rewrite the model output file
        obsdf.to_csv(obs_csv)
        # check 
        obsdf = pd.read_csv(obs_csv, index_col=0)
        assert (obsdf.index==check.index).all()
        return obsdf

    # update the SFR obs csv
    obs_srf = update_obs_csv(os.path.join(t_d,"sfr.csv"))
    # update the heads obs csv
    obs_hds = update_obs_csv(os.path.join(t_d,"heads.csv"))

    # %% [markdown]
    # OK...now we can run the postprocessing function to update the "tdiff" model output csv's. Copy across the `helpers.py` we used during the `PstFrom` tutorial. Then import it and run the `process_secondary_obs()` function.



    process_secondary_obs(ws=t_d)

    # %%
    # the oname column in the pst.observation_data provides a useful way to select observations in this case
    obs.oname.unique()

    # %%
    org_nnzobs = pst.nnz_obs
        #if (pst.nnz_obs-org_nnzobs)!=0:
        #    assert (pst.nnz_obs-org_nnzobs)==obs_data.count().sum()

    # %%
    print('Number of nonzero obs:', pst.nnz_obs)

    diff_obsdict = {'sfrtd': "sfr.tdiff.csv", 
                    'hdstd': "heads.tdiff.csv",
                    }

    for keys, value in diff_obsdict.items():
        print(keys)
        # get subselct of obs names
        obs_names = obs.loc[obs.oname.isin([keys]), 'obsnme']
        # get df
        obs_csv = pd.read_csv(os.path.join(t_d,value),index_col=0)
        # specify cols to use; make use of info recorded in pst.observation_data to only select cols with measured data
        usecols = list(set((map(str.upper, obs.loc[pst.nnz_obs_names,'usecol'].unique()))) & set(obs_csv.columns.tolist()))
        obs_csv = obs_csv.loc[:, usecols]
        # for checking
        org_nnz_obs_names = pst.nnz_obs_names
        # run the function
        update_pst_obsvals(obs_names,
                            obs_csv)
        # verify num of new nnz obs
        print(pst.nnz_obs)
        print(len(org_nnz_obs_names))
        print(len(usecols))
        assert (pst.nnz_obs-len(org_nnz_obs_names))==12*len(usecols), [i for i in pst.nnz_obs_names if i not in org_nnz_obs_names]

    # %%
    pst.nnz_obs_groups

    # %%
    pst.nnz_obs_names

    # %% [markdown]
    # The next cell does some sneaky things in the background to populate `obsvals` for forecast observations just so that we can keep track of the truth. In real-world applications you might assign values that reflect decision-criteria (such as limits at which "bad things" happen, for example) simply as a convenience. For the purposes of history matching, these values have no impact because they are assigned zero weight. They can play a role in specifying constraints when undertaking optimisation problems.  

    # %%
    pst.observation_data.loc[pst.forecast_names]

    # %%
    #hbd.prep_forecasts(pst)

    # %%
    pst.observation_data.loc[pst.forecast_names]

    # %% [markdown]
    # ## Observation Weights
    # 
    # Of all the issues that we have seen over the years, none is greater than (in)appropriate weighting strategies.  It is a critical and fundamental component of any inverse problem, but is especially important in settings where the model is imperfect simulator and the observation data are noisy and there are diverse types of data.  Goundwater modeling anyone? 
    # 
    # In essence the weights will change the shape of the objective function surface in parameter space, moving the minimum around and altering the path to the minimum (this can be seen visually in the response surface notebooks).  Given the important role weights play in the outcome of a history-matching/data assimilation analysis, rarely is a weighting strategy "one and done", instead it is continuously revisited during a modeling analysis, based on what happened during the previous history-matching attempt.  
    # 
    # We are going to start off by taking a look at our current objective function value and the relative contributions from the various observation groups - these relative contributions are a function of the residuals and weights in each group. Recall that this is the objective function value with **initial parameter values** and the default observations weights.
    # 
    # First off, we need to get PEST to run the model once so that the objective function can be calculated. Let's do that now. Start by reading the control file and checking that NOPTMAX is set to zero:
    # 
    # 

    # %%
    # check noptmax
    pst.control_data.noptmax

    # %% [markdown]
    # You got a zero? Alrighty then! Let's write the uprated control file and run PEST again and see what that has done to our Phi:

    # %%
    pst.write(os.path.join(t_d,pst_file),version=2)

    # %%
    pyemu.os_utils.run("pestpp-ies.exe {0}".format(pst_file),cwd=t_d)

    # %% [markdown]
    # Now we need to reload the `Pst` control file so that the residuals are updated:

    # %%
    pst = pyemu.Pst(os.path.join(t_d, pst_file))
    pst.phi

    # %% [markdown]
    # Jeepers - that's large! Before we race off and start running PEST to lower it we should compare simulated and measured values and take a look at the components of Phi. 
    # 
    # Let's start with taking a closer look. The `pst.phi_components` attribute returns a dictionary of the observation group names and their contribution to the overall value of Phi. 

    # %%
    pst.phi_components

    # %% [markdown]
    # 
    # Unfortunately, in this case we have too many observation groups to easily display (we assigned each individual time series to its own observation group; this is a default setting in `pyemu.PstFrom`). 
    # 
    # So let's use `Pandas` to help us summarize this information (note: `pyemu.plot_utils.res_phi_pie()` does the same thing, but it looks a bit ugly because of the large number of observation groups). To make it easier, we are going to just look at the nonzero observation groups:

    # %%
    nnz_phi_components = {k:pst.phi_components[k] for k in pst.nnz_obs_groups} # that's a dictionary comprehension there y'all
    nnz_phi_components

    # %% [markdown]
    # And while we are at it, plot these in a pie chart. 
    # 
    # If you wish, try displaying this with `pyemu.plot_utils.res_phi_pie()` instead. Because of the large number of columns it's not going to be pretty, but it gets the job done.

    # %%
    phicomp = pd.Series(nnz_phi_components)
    plt.pie(phicomp, labels=phicomp.index.values);
    #pyemu.plot_utils.res_phi_pie(pst,);

    # %% [markdown]
    # Well that is certainly not ideal - phi is dominated by the SFR observation groups. Why? Because the magnitude of these observation values are much larger than groundwater-level based observations, so we can expect the residuals in SFR observations to be yuge compared to groundwater level residuals...and we assigned the same weight to all of them...
    # 
    # Now we have some choices to make.  In many settings, there are certain observations (or observation groups) that are of increased importance, whether its for predictive reasons (like some data are more similar to the predictive outputs from the modeling) or political - "show me obs vs sim for well XXX"...If this is the case, then it is probably important to give those observations a larger portion of the composite objective function so that the results of the history matching better reproduce those important observations.  
    # 
    # In this set of notebooks, we will use another very common approach: give all observations groups an equal portion of the composite objective function.  This basically says "all of the different observation groups are important, so do your best with all of them"
    # 
    # The `Pst.adjust_weights()` method provides a mechanism to fine tune observation weights according to their contribution to the objective function. (*Side note: the PWTADJ1 utility from the PEST-suite automates this same process of "weighting for visibility".*) 
    # 
    # We start by creating a dictionary of non-zero weighted observation group names and their respective contributions to the objective function. Herein, we will use the existing composite phi value as the target composite phi...
    # 

    # %%
    # create a dictionary of group names and weights
    contrib_per_group = pst.phi / float(len(pst.nnz_obs_groups))
    balanced_groups = {grp:contrib_per_group for grp in pst.nnz_obs_groups}
    balanced_groups

    # %%
    # make all non-zero weighted groups have a contribution of 100.0
    pst.adjust_weights(obsgrp_dict=balanced_groups,)

    # %% [markdown]
    # Let's take a look at how that has affected the contributions to Phi:

    # %%
    plt.figure(figsize=(7,7))
    phicomp = pd.Series({k:pst.phi_components[k] for k in pst.nnz_obs_groups})
    plt.pie(phicomp, labels=phicomp.index.values);
    plt.tight_layout()

    # %% [markdown]
    # Better! Now each observation group contributes equally to the objective function

    # %% [markdown]
    # The next cell adds in a column to the `pst.observation_data` for checking purposes in subsequent tutorials. In practice, when you have lots of model outputs treated as "obserations" in the pest control file, setting a flag to indicate exactly which observation quantities correspond to actual real-world information can be important for tracking things through your workflow...

    # %%
    pst.observation_data.loc[pst.nnz_obs_names,'observed'] = 1

    # %% [markdown]
    # ### Understanding Observation Weights and Measurement Noise
    # 
    # Let's have a look at what weight values were assigned to our observation groups:

    # %%
    obs = pst.observation_data
    for group in pst.nnz_obs_groups:
        print(group,obs.loc[obs.obgnme==group,"weight"].unique())

    # %% [markdown]
    # Ok, some variability there, and, as expected, the sfr flowout observations have been given a very low weight and the groundwater level obs have been given a very high weight - this is simply to overcome the difference in magnitudes between these two different data types.  All good...or is it?
    # 
    # In standard deterministic parameter estimation, only the relative difference between the weights matters, so we are fine there...but in uncertainty analysis, we often want to account for the contribution from measurement noise and we haven't told any of the pest++ tools not to use the inverse of the weights to approximate measurement noise, and this is a problem because those weights we assigned have no relation to measurement noise!  This can cause massive problems later, especially is you are using explicit noise realizations in uncertainty analysis - Imagine how much SFR flow noise is implied by that tiny weight?  It's easy to see how negative SFR flow noise values might be drawn with that small of a weight (high of a standard deviation) #badtimes.   
    # 
    # So what can we do?  Well there are options.  An easy way is to simply supply a "standard_deviation" column in the `pst.observation_data` dataframe that will cause these values to be used to represent measurement noise.  

    # %%
    obs = pst.observation_data
    obs.loc[:,"standard_deviation"] = np.nan
    hds_obs = [o for o in pst.nnz_obs_names if "oname:hds_" in o]
    assert len(hds_obs) > 0
    obs.loc[hds_obs,"standard_deviation"] = 0.3
    hdstd_obs = [o for o in pst.nnz_obs_names if "oname:hdstd_" in o]
    assert len(hdstd_obs) > 0
    obs.loc[hdstd_obs,"standard_deviation"] = 0.001

    sfr_obs = [o for o in pst.nnz_obs_names if "oname:sfr_" in o]
    assert len(sfr_obs) > 0
    # here we will used noise that is a function of the observed flow value so that 
    # when flow is high, noise is high.
    obs.loc[sfr_obs,"standard_deviation"] = obs.loc[sfr_obs,"obsval"] * 0.15
    sfrtd_obs = [o for o in pst.nnz_obs_names if "oname:sfrtd_" in o]
    assert len(sfrtd_obs) > 0
    obs.loc[sfrtd_obs,"standard_deviation"] = obs.loc[sfrtd_obs,"obsval"] * 0.15

    # %%
    obs.loc[pst.nnz_obs_names,["obsval","standard_deviation"]]

    # %%
    pst.write(os.path.join(t_d,pst_file),version=2)
    return

if __name__ == "__main__":
    run_pstfrom()
    set_weights()
