pip install --upgrade -r requirements/requirements.txt
#pip install --upgrade -r requirements/sagemaker_requiremnets.txt

pip install awscli -U
apt update 
apt install less
pip install --force-reinstall typing-extensions==4.5.0
pip install spacy[cuda-113] -U
python -m spacy download en_core_web_trf
