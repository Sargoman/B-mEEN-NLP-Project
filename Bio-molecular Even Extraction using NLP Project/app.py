from flask import Flask,url_for,render_template,request,send_file,redirect,request
from flask_uploads import UploadSet,configure_uploads,ALL,DATA
from werkzeug import secure_filename

# Other Packages
import os
import pandas as pd

# NER package
import stanza


# Download and initialize a pipelines with various NER model
stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
stanza.download('en', package='craft', processors={'ner': 'bc5cdr'})
stanza.download('en', package='craft', processors={'ner': 'bionlp13cg'})

nlpMI = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'})
nlpCbc = stanza.Pipeline('en', package='craft', processors={'ner': 'bc5cdr'})
nlpCbio = stanza.Pipeline('en', package='craft', processors={'ner': 'bionlp13cg'})

# Function to capture all token_type with token in dataframe
def entityExtract(doc):
    x= []
    y= []
    for ent in doc.entities:
        x.append(ent.text)
        y.append(ent.type)
    
    df = pd.DataFrame()
    df['text'] = x
    df['type'] = y
    return df

# Funciton to capture specific token_type
def text_label(label, dfrm):
	text_label_li = []
	for i in range(len(dfrm['type'])):
		if dfrm['type'][i] == label.upper():
			text_label_li.append(dfrm['text'][i])
	return text_label_li    

# Capturing all TEST, PROBLEM, TREATMENT
def mi_opt_test(data):
	docMI = nlpMI(data)
	dfrm = entityExtract(docMI)
	rel = set(text_label('TEST', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def mi_opt_problem(data):
	docMI = nlpMI(data)
	dfrm = entityExtract(docMI)
	rel = set(text_label('PROBLEM', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def mi_opt_treatment(data):
	docMI = nlpMI(data)
	dfrm = entityExtract(docMI)
	rel = set(text_label('TREATMENT', dfrm))
	l_rel = len(rel)
	return rel, l_rel


# Capturing all DISEASE, CHEMICAL
def cbc_opt_chemical(data):
	docCbc = nlpCbc(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('DISEASE', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbc_opt_disease(data):
	docCbc = nlpCbc(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('CHEMICAL', dfrm))
	l_rel = len(rel)
	return rel, l_rel

# Capturing all DISEASE, CHEMICAL
def cbio_opt_amino_acid(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('AMINO_ACID', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_cancer(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('CANCER', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_gene(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('GENE_OR_GENE_PRODUCT', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_organ(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('ORGAN', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_organism(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('ORGANISM', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_tissue(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('TISSUE', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_simple_chemical(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('SIMPLE_CHEMICAL', dfrm))
	l_rel = len(rel)
	return rel, l_rel

def cbio_opt_multi_tissue_structure(data):
	docCbc = nlpCbio(data)
	dfrm = entityExtract(docCbc)
	rel = set(text_label('MULTI-TISSUE_STRUCTURE', dfrm))
	l_rel = len(rel)
	return rel, l_rel




# Initialize App
app = Flask(__name__)

# Configuration For Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadedfiles'
configure_uploads(app,files)


# Home 
@app.route('/')
def index():

	return render_template("index.html")


# File Upload & Extract Entities
@app.route('/extract', methods=['GET', 'POST'])
def extract():
	if request.method == 'POST' and 'rawtext' in request.files:
		file = request.files['rawtext']
		filename = secure_filename(file.filename)
		file.save(os.path.join('static/uploadedfiles', filename))

		with open(os.path.join('static/uploadedfiles',filename), 'r+', encoding="utf-8") as f:
			c_text = f.read()

		mite,mitel = mi_opt_test(c_text)
		mitr, mitrl = mi_opt_treatment(c_text)
		mipr, miprl = mi_opt_problem(c_text)
		cbcd, cbcdl = cbc_opt_disease(c_text)
		cbcc, cbccl = cbc_opt_chemical(c_text)
		cbioc, cbiocl = cbio_opt_cancer(c_text)
		cbioaa, cbioaal = cbio_opt_amino_acid(c_text)
		cbiot, cbiotl = cbio_opt_tissue(c_text)
		cbioo, cbiool = cbio_opt_organism(c_text)
		cbiog, cbiogl =  cbio_opt_gene(c_text)
		cbioog, cbioogl =  cbio_opt_organ(c_text)
		cbiosc, cbioscl =  cbio_opt_simple_chemical(c_text)
		cbiomts, cbiomtsl =  cbio_opt_multi_tissue_structure(c_text)






	return render_template("index.html",cbiomts=cbiomts,cbiomtsl=cbiomtsl,cbiosc=cbiosc,cbioscl=cbioscl,cbioog=cbioog, cbioogl=cbioogl,cbiog=cbiog,cbiogl=cbiogl,cbioo=cbioo,cbiool=cbiool,cbiot=cbiot,cbiotl=cbiotl,cbioaa=cbioaa,cbioaal=cbioaal,cbioc=cbioc,cbiocl=cbiocl,mite=mite,mitel=mitel,mitr=mitr,mitrl=mitrl,mipr=mipr,miprl=miprl,cbcd=cbcd,cbcdl=cbcdl,cbcc=cbcc,cbccl=cbccl)





if __name__ == '__main__':
	app.run(debug=True)