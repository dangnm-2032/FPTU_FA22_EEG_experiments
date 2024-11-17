import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = os.path.normpath(os.getcwd()).split(os.path.sep)[-1]

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/conponents/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "config/params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "experiments/trials.py",
    "setup.py",
    "tests/tests.py",
    "Makefile",
]

for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            logging.info(f"Creating empty file: {filepath}")
            pass
    else:
        logging.info(f"{filename} already exists")

with open(f"src/{project_name}/config/configuration.py", 'w') as f:
    f.write("""from pathlib import Path
from vinewschatbot.constants import *
from vinewschatbot.utils import *
from vinewschatbot.entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)""")
    
with open(f"src/{project_name}/config/__init__.py", 'w') as f:
    f.write("from .configuration import *")

with open(f"src/{project_name}/logging/__init__.py", 'w') as f:
    f.write("""import os 
import sys 
import logging

logging_str = "%(asctime)s: %(levelname)s: %(module)s: %(message)s"
log_dir = "./logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
print(os.getcwd())
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
   level=logging.INFO, 
   format=logging_str, 
   
   handlers=[
       logging.FileHandler(log_filepath),
       logging.StreamHandler(sys.stdout)
   ]
)
logger = logging.getLogger("Vi-News-Chatbot")""")
    
with open(f"src/{project_name}/entity/__init__.py", 'w') as f:
    f.write('''from dataclasses import dataclass
from pathlib import Path''')
    
with open(f"src/{project_name}/constants/__init__.py", 'w') as f:
    f.write('''from pathlib import Path
import os

CONFIG_FILE_PATH = Path('config/config.yaml')
PARAMS_FILE_PATH = Path('config/params.yaml')

CURRENT_WORKING_DIRECTORY = os.getcwd()''')
    
with open('setup.py', 'w') as f:
    f.write('''import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = ""
AUTHOR_USER_NAME = "dangnm-2032"
SRC_REPO = ""
AUTHOR_EMAIL = "dangnm.working@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)''')
    
with open('.gitignore', 'w') as f:
    f.write('''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/#use-with-ide
.pdm.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/
/artifacts
/logs
/wandb''')
    
with open(f"src/{project_name}/utils/common.py", 'w') as f:
    f.write('''import yaml
from pathlib import Path
from box import ConfigBox 
from box.exceptions import BoxValueError
from FPTU_FA24_EEG_Artifacts_Recognition.logging import logger
from ensure import ensure_annotations
import os
from FPTU_FA24_EEG_Artifacts_Recognition.constants import *

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox: 
    """Read yaml file and returns ConfigBox instance

    Args:
        path_to_yaml: Path to yaml file
    
    Releases:
        ValueError: if yaml file is empty 
        e: empty file
    Returns:
        ConfigBox instance
    """
    try: 
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file '{path_to_yaml}' loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file: '{path_to_yaml}' is empty")
    except Exception as e: 
        raise e
    
def stage_name(
    text: str
) -> str:
    text = ">" * 10 + text + "<" * 10
    return text''')