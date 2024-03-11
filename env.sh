#echo "Start to create the Conda virtual environment... ..."
#conda create -n ed2 python==3.10 -y
#source activate ed2
#
#echo "Start to install packages... ..."
#pip install -r requirements.txt

echo "Start to install orbkit... ..."
cd $HOME
git clone https://github.com/orbkit/orbkit.git
cd orbkit
git checkout cc17072
python setup.py build_ext --inplace clean
