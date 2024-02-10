from kaggle_environments import make
from IPython.display import HTML, display
import time

env = make("connectx", debug=True)
env.run(["random", "random"])
env.render(mode="ipython", width=500, height=450)


