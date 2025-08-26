import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import seaborn as sns
import  matplotlib.patheffects  as PathEffects
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score,roc_auc_score
from sklearn import svm
from sklearn.decomposition import PCA
test_data= r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis_gen'
data_set = '\\Proposed'
path_infor = r'/data1/sxr/fmri_dti_synthesis/data_scan30/csv/DFC_CLINICAL/HC_SCD_MCI_AD/val_fold1.csv'
Source_fmri= []
Gen_fmri = []
Source_dti = []
Gen_dti = []
RS=123
label =[]
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})
def read_nii(path):
    img_pil_test = nib.load(path)
    img_arr_test = np.array(img_pil_test.get_fdata())
    return img_arr_test
with open(path_infor, 'r') as csvfile:
    reader = csv.reader(csvfile)
    files_label_1 = [row[2] for row in reader]
with open(path_infor, 'r') as csvfile:
    reader = csv.reader(csvfile)
    files_label_2 = [row[3] for row in reader]
with open(path_infor, 'r') as csvfile:
    reader = csv.reader(csvfile)
    files_label_3 = [row[4] for row in reader]
with open(path_infor, 'r') as csvfile:
    reader = csv.reader(csvfile)
    files_label_4 = [row[5] for row in reader]
label_hot = np.array([files_label_1,files_label_2,files_label_3,files_label_4])

for j in range(label_hot.shape[1]):
    label_j_hot = label_hot[:,j]
    if (np.array(['1', '0', '0', '0']) == label_j_hot).all():
        label.append(0)
    # elif  (np.array(['0', '1', '0', '0']) == label_j_hot).all():
    #     label.append(1)
    elif (np.array(['0', '0', '1', '0']) == label_j_hot).all():
        label.append(1)
    elif (np.array(['0', '0', '0', '1']) == label_j_hot).all():
        label.append(2)
    else:
        print('Wrong label')

#print(label)

def hard2onehot(label):
      #print(label.shape)
      onehot_arr = np.array([1,0,0])
      for i in range(len(label)):
            if label[i] == 0:
               tmp_onehot = np.array([1,0,0])

               onehot_arr = np.concatenate((onehot_arr,tmp_onehot),axis=0)
            if label[i] == 1: 
               tmp_onehot = np.array([0,1,0])

               onehot_arr = np.concatenate((onehot_arr,tmp_onehot),axis=0)
            if label[i] == 2:
               tmp_onehot = np.array([0,0,1])

               onehot_arr = np.concatenate((onehot_arr,tmp_onehot),axis=0)

      onehot_arr = onehot_arr.reshape(int(len(onehot_arr)/3),3)
      #print(onehot_arr)
      return onehot_arr
def fashion_scatter(x,colors):
      num_class = len(np.unique(colors))
      palette = np.array(sns.color_palette('hls', num_class))
      f=plt.figure(figsize = (8,8))
      ax = plt.subplot(aspect='equal')
      sc = ax.scatter(x[:,0],x[:,1], lw=0, s=40, c=palette[colors.astype(int)])
      plt.xlim(-25,25)
      plt.ylim(-25,25)

      ax.axis('off')
      ax.axis('tight')
      txts =[]
      handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
                            markersize=10, markerfacecolor=palette[i]) for i in range(num_class)]
      ax.legend(handles=handles, loc='upper right', title='Classes')
      return f,ax,sc,txts

for i in range(0,173):

    gen_dti_name ='/gen_sample%d_f2d.nii'%i
    gen_fmri_name ='/gen_sample%d_d2f.nii'%i
    tar_dti_name ='/tar_iter%d_f2d.nii'%i
    tar_fmri_name ='/tar_iter%d_d2f.nii'%i
    path_fMRI = test_data+gen_fmri_name
    path_dti = test_data+gen_dti_name
    path_tar_dti = test_data+ tar_dti_name
    path_tar_fmri = test_data+  tar_fmri_name
    Gen_dti.append(read_nii(path_dti).flatten())
    Gen_fmri.append(read_nii(path_fMRI).flatten())
    Source_dti.append(read_nii(path_tar_dti).flatten())
    Source_fmri.append(read_nii(path_tar_fmri).flatten())

'''
PCA reduce dimension
'''
PCA_dimension = 150
PCA_gen_dti = PCA(n_components=PCA_dimension)
pca_result_gen_dti = PCA_gen_dti.fit_transform(np.array(Gen_dti))
PCA_source_dti = PCA(n_components=PCA_dimension)
pca_result_source_dti = PCA_source_dti.fit_transform(np.array(Source_dti))
PCA_gen_fmri = PCA(n_components=PCA_dimension)
pca_result_gen_fmri = PCA_gen_fmri.fit_transform(np.array(Gen_fmri))
PCA_source_fmri = PCA(n_components=PCA_dimension)
pca_result_source_fmri =PCA_source_fmri.fit_transform(np.array(Source_fmri))

data_all_dti=  np.concatenate((pca_result_gen_dti,pca_result_source_dti),axis=0)
# gen_dti_label = np.array('Gen DTI' + i for i in np.array(label))
label_all = np.concatenate([np.array(label),np.array(label)+3])
data_all_fmri=  np.concatenate((  \
                           pca_result_gen_fmri, pca_result_source_fmri),axis=0)
# gen_dti_label = np.array('Gen DTI' + i for i in np.array(label))




'''
SVM classification
'''
data_genDTI_resourcefMRI = np.concatenate((pca_result_gen_dti,pca_result_source_fmri),axis=1)
data_genDTI_genfMRI = np.concatenate((pca_result_gen_dti,pca_result_gen_fmri),axis=1)
data_resourceDTI_genfMRI= np.concatenate((pca_result_source_dti,pca_result_gen_fmri),axis=1)
data_resourceDTI_resourcefMRI= np.concatenate((pca_result_source_dti,pca_result_source_fmri),axis=1)
'''
gen DTI and resource fMRI build model, real data test
'''
clf_genDTI_resourcefMRI = svm.SVC(kernel="linear", C=1.0,decision_function_shape='ovo')
clf_genDTI_resourcefMRI.fit(data_genDTI_resourcefMRI,np.array(label))
pred_label_genDTI_resourcefMRI =clf_genDTI_resourcefMRI.predict(data_resourceDTI_resourcefMRI)
print('Gen DTI prediction ACC: %f  PRE: %f Recall: %f F1: %f AUC: %f'%(
    accuracy_score(np.array(label),pred_label_genDTI_resourcefMRI),
    precision_score(np.array(label),pred_label_genDTI_resourcefMRI,average='macro'),
    recall_score(np.array(label),pred_label_genDTI_resourcefMRI,average='macro'),
    f1_score(np.array(label),pred_label_genDTI_resourcefMRI,average='macro'),
    roc_auc_score(hard2onehot(np.array(label)), hard2onehot(pred_label_genDTI_resourcefMRI),average='macro', multi_class='ovo')
)
      )

print(hard2onehot(np.array(label)).shape)

'''
gen fMRI and resource DTI build model, real data test
'''
clf_resourceDTI_genfMRI = svm.SVC(kernel="linear", C=1.0,decision_function_shape='ovo')
clf_resourceDTI_genfMRI.fit(data_resourceDTI_genfMRI,np.array(label))
pred_label_resourceDTI_genfMRI =clf_resourceDTI_genfMRI.predict(data_resourceDTI_resourcefMRI)
print('Gen fMRI prediction ACC: %f  PRE: %f Recall: %f F1: %f AUC: %f' % (
    accuracy_score(np.array(label), pred_label_resourceDTI_genfMRI),
    precision_score(np.array(label), pred_label_resourceDTI_genfMRI,average='macro'),
    recall_score(np.array(label), pred_label_resourceDTI_genfMRI,average='macro'),
    f1_score(np.array(label), pred_label_resourceDTI_genfMRI,average='macro'),
    roc_auc_score(hard2onehot(np.array(label)), hard2onehot(pred_label_resourceDTI_genfMRI),average='macro', multi_class='ovo')
 )
      )

'''
gen fMRI and gen DTI build model, real data test
'''
clf_genDTI_genfMRI = svm.SVC(kernel="linear", C=1.0,decision_function_shape='ovo')
clf_genDTI_genfMRI.fit(data_genDTI_genfMRI,np.array(label))
pred_label_genDTI_genfMRI =clf_genDTI_genfMRI.predict(data_resourceDTI_resourcefMRI)
print('Gen fMRI and gen DTI prediction \n ACC: %f  PRE: %f Recall: %f F1: %f AUC: %f' % (
    accuracy_score(np.array(label), pred_label_genDTI_genfMRI),
    precision_score(np.array(label), pred_label_genDTI_genfMRI,average='macro'),
    recall_score(np.array(label), pred_label_genDTI_genfMRI,average='macro'),
    f1_score(np.array(label), pred_label_genDTI_genfMRI,average='macro'),
  roc_auc_score(hard2onehot(np.array(label)), hard2onehot(pred_label_genDTI_genfMRI),average='macro', multi_class='ovo')
)
      )

'''
True fMRI and resource DTI build model, gen data test
'''
clf_resourceDTI_resourcefMRI = svm.SVC(kernel="linear", C=1.0,decision_function_shape='ovo')
clf_resourceDTI_resourcefMRI.fit(data_resourceDTI_resourcefMRI,np.array(label))
pred_label_resourceDTI_resourcefMRI_1 =clf_resourceDTI_resourcefMRI.predict(data_resourceDTI_genfMRI)
print('Resource fMRI and Resource DTI 1 \n prediction ACC: %f  PRE: %f Recall: %f F1: %f AUC: %f' % (
    accuracy_score(np.array(label), pred_label_resourceDTI_resourcefMRI_1),
    precision_score(np.array(label), pred_label_resourceDTI_resourcefMRI_1,average='macro'),
    recall_score(np.array(label), pred_label_resourceDTI_resourcefMRI_1,average='macro'),
    f1_score(np.array(label), pred_label_resourceDTI_resourcefMRI_1,average='macro'),
    roc_auc_score(hard2onehot(np.array(label)), hard2onehot(pred_label_resourceDTI_resourcefMRI_1),average='macro', multi_class='ovo')
)
      )



pred_label_resourceDTI_resourcefMRI_2 =clf_resourceDTI_resourcefMRI.predict(data_genDTI_resourcefMRI)
print('Resource fMRI and Resource prediction 2 ACC: %f  PRE: %f Recall: %f F1: %f AUC:%f' % (
    accuracy_score(np.array(label), pred_label_resourceDTI_resourcefMRI_2),
    precision_score(np.array(label), pred_label_resourceDTI_resourcefMRI_2,average='macro'),
    recall_score(np.array(label), pred_label_resourceDTI_resourcefMRI_2,average='macro'),
    f1_score(np.array(label), pred_label_resourceDTI_resourcefMRI_2,average='macro'),
    roc_auc_score(hard2onehot(np.array(label)), hard2onehot(pred_label_resourceDTI_resourcefMRI_2),average='macro', multi_class='ovo')
)
      )

pred_label_resourceDTI_resourcefMRI_3 =clf_resourceDTI_resourcefMRI.predict(data_genDTI_genfMRI)
print('Resource fMRI and Resource prediction 3 ACC: %f  PRE: %f Recall: %f F1: %f AUC :%f' % (
    accuracy_score(np.array(label), pred_label_resourceDTI_resourcefMRI_3),
    precision_score(np.array(label), pred_label_resourceDTI_resourcefMRI_3,average='macro'),
    recall_score(np.array(label), pred_label_resourceDTI_resourcefMRI_3,average='macro'),
    f1_score(np.array(label), pred_label_resourceDTI_resourcefMRI_3,average='macro'),
     roc_auc_score(hard2onehot(np.array(label)), hard2onehot(pred_label_resourceDTI_resourcefMRI_3),average='macro', multi_class='ovo')

)
      )

'''
TSNE
'''
fashion_tsne_dti = TSNE(random_state=RS,n_jobs=-1).fit_transform(data_all_dti)
f_all_dti, ax_all_dti,sc_all_dti, txts_all_dti= fashion_scatter(fashion_tsne_dti,label_all)
#f_all_dti.show()

fashion_tsne_fmri = TSNE(random_state=RS,n_jobs=-1).fit_transform(data_all_fmri)
f_all_fmri, ax_all_fmri,sc_all_fmri, txts_all_fmri= fashion_scatter(fashion_tsne_fmri,label_all)
#f_all_fmri.show()
# sns.scatterplot(fashion_tsne_[:,0], fashion_tsne_[:,1], hue=label_all, legend='full', palette=palette)
# fashion_tsne_gen_dti = TSNE(random_state=RS,n_jobs=-1).fit_transform(pca_result_gen_dti)
# fashion_tsne_gen_fmri = TSNE(random_state=RS,n_jobs=-1).fit_transform(pca_result_gen_fmri)
# fashion_tsne_source_fmri = TSNE(random_state=RS,n_jobs=-1).fit_transform(pca_result_source_fmri)
# fashion_tsne_source_dti= TSNE(random_state=RS,n_jobs=-1).fit_transform(pca_result_source_dti)
#
# f_gen_dti, ax_gen_dti,sc_gen_dti, txts_gen_dti= fashion_scatter(fashion_tsne_gen_dti,np.array(label))
# f_source_dti, ax_source_dti, sc_source_dti,txts_souce_dti = fashion_scatter(fashion_tsne_source_dti,np.array(label))
# f_gen_fmri, ax_gen_fmri, sc_gen_fmri,txts_gen_fmri = fashion_scatter(fashion_tsne_gen_fmri,np.array(label))
# f_source_fmri, ax_source_fmri, sc_source_fmri, txts_source_fmri = fashion_scatter(fashion_tsne_source_fmri,np.array(label))
# f_gen_dti.show()
# f_source_dti.show()
#
# f_gen_fmri.show()
# f_source_fmri.show()
