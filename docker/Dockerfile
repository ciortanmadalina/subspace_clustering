# docker build -t discovers .
# docker run -it --runtime=nvidia -v ~/notebooks:/tf/notebooks -p 8888:8888 discovers
# FROM tensorflow/tensorflow:1.15.2-py3-jupyter # uncomment this when running on cpu
FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

RUN pip install tqdm \
    keras==2.3.1 \
    pandas \
    seaborn \
    xlrd==1.2.0 \
    statsmodels \
    xgboost \
    sklearn \
    openpyxl \
    gseapy \
    scanpy==1.5.1 \
    lifelines \
    leidenalg  \
    glob2 \
    xlrd==1.2.0 \
    Pillow
RUN pip uninstall -y enum34
RUN pip install hdbscan
        
        
# Jupyter notebook configuration
RUN pip install yapf \
    jupyter_contrib_nbextensions \
    jupyter_highlight_selected_word

RUN jupyter contrib nbextension install --user
RUN jupyter nbextension install https://github.com/jfbercher/code_prettify/archive/master.zip --user
RUN jupyter nbextension enable code_prettify-master/code_prettify
RUN jupyter nbextension install --py jupyter_highlight_selected_word
RUN jupyter nbextension enable highlight_selected_word/main

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
