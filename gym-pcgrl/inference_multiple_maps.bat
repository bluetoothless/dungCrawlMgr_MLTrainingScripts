@echo off
set conda_dir=D:\Anaconda3
set conda_env_name=pcgrl
call %conda_dir%\Scripts\conda activate %conda_env_name%
FOR /L %%G IN (1,1,50) DO (
    python .\inference.py
)
