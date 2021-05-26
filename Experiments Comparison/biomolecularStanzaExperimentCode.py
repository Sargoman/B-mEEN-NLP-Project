# -*- coding: utf-8 -*-
"""
Created on Wed May 26 06:54:56 2021

@author: SAURAV GUPTA
"""

# NER package
import stanza

# Other packages
from collections import Counter
import pandas as pd


# Download and initialize a pipelines with various NER model
stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
stanza.download('en', package='craft', processors={'ner': 'bc5cdr'})
stanza.download('en', package='craft', processors={'ner': 'bionlp13cg'})

nlpMI = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'})
nlpCbc = stanza.Pipeline('en', package='craft', processors={'ner': 'bc5cdr'})
nlpCbio = stanza.Pipeline('en', package='craft', processors={'ner': 'bionlp13cg'})

# annotate clinical text
docx = """Vascular endothelial growth factor-B and vascular endothelial growth factor-C expression in renal cell carcinomas: regulation by the von Hippel-Lindau gene and hypoxia.

Angiogenesis is essential for tumor growth and metastasis. It is regulated by numerous angiogenic factors, one of the most important being vascular endothelial growth factor (VEGF). Recently VEGF-B and VEGF-C, two new VEGF family members, have been identified that bind to the tyrosine kinase receptors flt-1 (VEGFR1), KDR (VEGFR2), and flt-4 (VEGFR3). Although the importance of VEGF-A has been shown in renal carcinomas, the contribution of these new ligands in kidney tumors is not clear. We have, therefore, measured the mRNA level of VEGF-B and VEGF-C together with their receptors by RNase protection assay (RPA) in 26 normal kidney samples and 45 renal cell cancers. We observed a significant up-regulation of VEGF-B (P = 0.002) but not VEGF-C (P = 0.3) in neoplastic kidney compared with normal tissues. In addition, although VEGF receptors were higher in tumors than normal kidney, there was a significant up-regulation of only flt-1 (P = 0.003) but not KDR (P = 0.12) or flt-4 (P = 0.09). There was also a significant correlation between VEGF-C and both of its receptors flt-4 (P = 0.006) and KDR (P = 0.03) but no association between VEGF-B and its receptor flt-1 (P = 0.23). A significant increase was observed in flt-1 (P  less than  0.001), KDR (P = 0.02), and flt-4 (P = 0.01) but not VEGF-B (P = 0.82) or VEGF-C (P = 0.52) expression in clear cell compared with chromophil (papillary) carcinomas. No significant association was demonstrated between VEGF-B, VEGF-C, flt-1, KDR, and flt-4 with patient sex, patient age, or tumor size (P  greater than  0.05). The effect of von Hippel-Lindau (VHL) gene and hypoxia on VEGF-B and VEGF-C expression in the renal carcinoma cell line 786-0 transfected with wild-type and mutant VHL was determined by growing cells under 21% O2- and 0.1% O2. In wild-type VHL cells, whereas VEGF-A was significantly up-regulated under hypoxic compared with normoxic conditions (P  less than  0.001), expression of VEGF-C was reduced (P  less than  0.002). Nevertheless, the repression of VEGF-C was lost in mutant VHL cell lines under hypoxia. In contrast VEGF-B was not regulated by VHL despite clear up-regulation in vivo. These findings strongly support an enhanced role for this pathway in clear cell carcinomas by regulating angiogenesis and/or lymphangiogenesis. The study shows that clear cell tumors are able to up-regulate angiogenic growth factor receptors more efficiently than chromophil (papillary), that clear cell tumors can use pathways independent of VHL to regulate angiogenesis, and that this combined regulation may account for their more aggressive phenotype, which suggests that targeting VEGFR1 (flt-l) may be particularly effective in these tumor types.

"""
# function to capture all token_type with token in dataframe
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

# funciton to capture specific token_type
def text_label(label, dfrm):
	text_label_li = []
	for i in range(len(dfrm['type'])):
		if dfrm['type'][i] == label.upper():
			text_label_li.append(dfrm['text'][i])
	return text_label_li    

# capturing all entity type tokens and number of tokens
def mi_opt_test(data):
	docMI = nlpMI(data)
	dfrm = entityExtract(docMI)
	rel = set(text_label('TEST', dfrm))
	l_rel = len(rel)
	return rel, l_rel


a, b = mi_opt_test(docx)

# manually calling each function
docMI = nlpMI(docx)
docCbc = nlpCbc(docx)
docCbio = nlpCbio(docx)

m = entityExtract(docMI)
n = entityExtract(docCbc)
o = entityExtract(docCbio)


Counter(m['type'])
Counter(n['type'])
Counter(o['type'])
