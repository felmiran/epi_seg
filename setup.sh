conda env update -f environment.yml -n episeg
source activate episeg
pip install -r requirements.txt
python -m ipykernel install --user --name episeg --display-name "Python (episeg)"

