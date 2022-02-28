# Sleep Wellth

Visit our product [here](https://fin4719-project-1.herokuapp.com)!

Sleep Wellth provides two different products. PaGrowth, our robo-advisor for passive investments, and QuantiFi, which aims to be the accredited investor’s first foray into quantitative trading strategies.

## PaGrowth
![image](https://user-images.githubusercontent.com/41572120/156004576-31495410-ee3d-4fa3-b8ad-31a9a3411956.png)

## QuantiFi
![image](https://user-images.githubusercontent.com/41572120/156004674-79c10c9b-8d8a-4b19-a14a-97ac4a4511b5.png)

# Developer's Guide

### Set up
After cloning the repo, enter:  
`pipenv shell`  
`streamlit run app.py`

### Installing new packages
`pipenv install <package name>`  

### Adding new pages 
1. In a new Python file, write all streamlit functions under `def display()`  
2. Import the page to `app.py`, and add the page under the `pages` dictionary.
