# causal-optoconnectics
Obtaining causal connectivity from optogenetic stimulation and electrical recordings

# Download
Code is tracked with git

Data is tracked with git-lfs

# Installation
To run the simulaitons you need to install NEST >= 2.8.

Then we recomend using virtualenvs and install with pip
```pip install -r requirements.txt```

# Procedings
Simulation data for e.g. `params_1` was generated by
```bash
python simulator.py params_1
python run_analysis.py params_1
```
Then the analysis to produce the figures is contained in respective jupyter notebooks

Experimental data was analysed in a dedicated docker environment.
If you want to run the container, you need to specify the location of your local copy of the causal-optogenetics repo
```bash
docker run -it -p 8888:8888 -v /path/to/causal-optoconnectics/:/home/jovyan/work/instrumentalVariable/causal-optoconnectics/ tristanstoeber/instrumentalvariable jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
After starting, copy the returned link with the complete jupyter token into your browser.
You need to change the docker container id to 'localhost'.

The output looks like this:
http://DockerContainerId:8888/?token=SomeJupyterToken

Experimental data is downloaded to the parent directory.


# Notes
serif in inkscape:
`sudo apt install fonts-cmu`
