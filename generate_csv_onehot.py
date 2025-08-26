import csv
import numpy as np
import os
from math import ceil
from utils import OsJoin
from sklearn.model_selection import KFold
from opts import parse_opts
import pandas as pd
import fnmatch
opt = parse_opts()
root = opt.data_root_path

#health_dir = opt.data_type + '_healthy'
#MCI_dir = opt.data_type + '_MCI'
csv_save_dir = OsJoin('csv/', opt.data_type, opt.category)
test_ratio = 0.0001
n_fold = 1000

Subs=list()
least_subs_name=r'/home/b23sxr/fmri_dti_synthesis/data_scan/Mean_Diffusion'
def search_session(sub, files_list):
    index = list()
    for i in range(len(files_list)):
        if sub in files_list[i]:
            index.append(i)
        else:
            continue
    return index
def search_near_session(sub_fea, session_index, session_names):
    Current_ses = sub_fea.split('_d')[1]
    if '_diffusion' in sub_fea:
        ses_cur_number = Current_ses
    if '_T1' in sub_fea:
        ses_cur_number = Current_ses.split('_T1')[0]
    if '_rsfMRI' in sub_fea:
        ses_cur_number = Current_ses.split('_rsfMRI')[0]
    index = session_index[0]
    gap = 1000
    for i in range(len(session_index)):
        session_name_total = session_names[session_index[i]]
        session_name_num = session_name_total .split('_UDSb4_d')[1]
        gap_tmp = abs(int(session_name_num) - int(ses_cur_number))
        if gap_tmp < gap:
            gap = gap_tmp
            index = session_index[i]
    return index

for filename in os.listdir(least_subs_name):
    if '_diffusion' in filename:
        sub = filename.split('_diffusion')[0]
        if 'Hospital' in sub:
            sub_new = sub.split('Hospital_')[1]
        if 'ADNI' in sub:
            sub_new = sub.split('ADNI_')[1]
        if 'OASIS' in sub:
            sub_new_OA = sub.split('OASIS-3_')[1]
            # sub_new = sub_new_OA.split('_')[0]
            sub_new = sub_new_OA
        Subs.append(sub_new)
    else:
      sub = filename.split('zfc_Covswra_')[1].split('_rsfMRI_timeseries_Dosenbach164.mat')[0]
      # sub_name_delPRE = sub.split('_PRE')[0]
      Subs.append(sub)
sub_info_excel_ADNI = pd.read_csv(r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan/Label_information/ADNI.csv', \
                                   usecols=[1, 2])
sub_info_excel_OASIS = pd.read_csv(r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan/Label_information/OASIS3.csv', \
                                   usecols=[0,1,4,18])
data_health = []
label_health = []
data_MCI = []
label_MCI = []
data_SCD = []
label_SCD = []
data_AD = []
label_AD = []
fea_num = 1
CN=0
OASIS_list =list()
# sub_num=len(Subs)
# data_health=np.empty([sub_num, fea_i])
# data_MCI=np.empty([sub_num, fea_i])
# data_SCD=np.empty([sub_num, fea_i])
# sub_n=0
HC_num = 0
MCI_num = 0
SCD_num = 0
AD_num = 0
subs_had = list()
for sub in Subs:
    if sub in subs_had:
        continue
    files_list = list()
    for feature in os.listdir(root):
        if 'csv' in feature:
            csv_contain=True
            continue
        # if 'Function' in feature:
        #     print(feature)
        for sub_fea in os.listdir(os.path.join(root, feature)):
#or str(sub_name_delPRE)+'_PRE' in str(sub_fea)
            if str(sub) in str(sub_fea):
                print(sub_fea)
                if sub_fea in files_list:
                    continue
                subs_had.append(sub)
                if 'diffusion' in sub_fea:
                     sub_fea_fmri = sub_fea.replace('diffusion','rsfMRI')
                elif 'rsfMRI' in sub_fea:
                     sub_fea_fmri = sub_fea.replace('rsfMRI','diffusion')
                files_list.append(sub_fea_fmri)
                files_list.append(sub_fea)
                if 'Hospital' in sub_fea:
                    if 'HC' in sub_fea and 'HC' in opt.category:
                        data_health.append(OsJoin(root, feature, sub_fea))
                        path_diff = os.path.join(root, feature, sub_fea)
                        if 'diffusion' in sub_fea:
                           path_fMRI = path_diff.replace('diffusion','rsfMRI').replace('Mean_Diffusion','Mean_Function')
                        elif 'rsfMRI' in sub_fea:
                            path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                         'Mean_Diffusion')
                        data_health.append(path_fMRI)
                        if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                            label_health.append([1,0,0,0])
                        elif opt.n_classes == 3 and opt.category == 'HC_SCD_MCI':
                            label_health.append([1,0,0])
                        elif opt.n_classes == 3 and opt.category == 'HC_MCI_AD':
                            label_health.append([1,0,0])
                        elif opt.n_classes == 2 and opt.category == 'HC_MCI':
                            label_health.append([1,0])
                        elif opt.n_classes == 2 and opt.category == 'HC_SCD':
                            label_health.append([1, 0])
                        elif opt.n_classes == 2 and opt.category == 'HC_AD':
                            label_health.append([1, 0])
                        HC_num = HC_num+1
                    elif 'MCI' in sub_fea and 'MCI' in opt.category:
                        data_MCI.append(OsJoin(root, feature, sub_fea))
                        path_diff = os.path.join(root, feature, sub_fea)
                        if 'diffusion' in sub_fea:
                           path_fMRI = path_diff.replace('diffusion','rsfMRI').replace('Mean_Diffusion','Mean_Function')
                        elif 'rsfMRI' in sub_fea:
                            path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                             'Mean_Diffusion')
                        data_MCI.append(path_fMRI)
                        if opt.n_classes == 4 and 'HC_SCD_MCI_AD' == opt.category:
                            label_MCI.append([0, 0, 1, 0])
                        elif opt.n_classes == 3 and 'HC_SCD_MCI' == opt.category:
                            label_MCI.append([0, 0, 1])
                        elif opt.n_classes == 3 and 'HC_MCI_AD' == opt.category:
                            label_MCI.append([0, 1, 0])
                        elif opt.n_classes == 2 and 'HC_MCI' == opt.category:
                            label_MCI.append([0, 1])
                        elif opt.n_classes == 2 and 'MCI_SCD' == opt.category:
                            label_MCI.append([0, 1])
                        elif opt.n_classes == 2 and 'MCI_AD' == opt.category:
                            label_MCI.append([1, 0])
                        MCI_num = MCI_num+1
                    elif 'SCD' in sub_fea and 'SCD' in opt.category:
                        # data_SCD.append(OsJoin(root, feature, sub_fea))
                        data_SCD.append(OsJoin(root, feature, sub_fea))
                        path_diff = os.path.join(root, feature, sub_fea)
                        if 'diffusion' in sub_fea:
                           path_fMRI = path_diff.replace('diffusion','rsfMRI').replace('Mean_Diffusion','Mean_Function')
                        elif 'rsfMRI' in sub_fea:
                            path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                         'Mean_Diffusion')
                        data_SCD.append(path_fMRI)
                        SCD_num = SCD_num+1
                        if opt.n_classes == 4 and 'HC_SCD_MCI_AD' == opt.category:
                            label_SCD.append([0, 1, 0, 0])
                        elif opt.n_classes == 3 and 'HC_SCD_AD' == opt.category:
                            label_SCD.append([0, 1, 0])
                        elif opt.n_classes == 3 and 'HC_SCD_MCI' == opt.category:
                            label_SCD.append([0, 1, 0])
                        elif opt.n_classes == 2 and 'HC_SCD' == opt.category:
                            label_SCD.append([0, 1])
                        elif opt.n_classes == 2 and 'SCD_MCI' == opt.category:
                            label_SCD.append([1, 0])
                        elif opt.n_classes == 2 and 'SCD_AD' == opt.category:
                            label_SCD.append([1, 0])
                    # else:
                    #     print(sub_fea)
                elif 'ADNI' in sub_fea: #data from adni
                    if str(sub) in str(sub_fea):
                        sub_index = list(sub_info_excel_ADNI['Subject']).index(sub)
                        sub_category = sub_info_excel_ADNI['Group'][sub_index]
                        if 'CN' in sub_category and ('CN' in opt.category or 'HC' in opt.category ):
                                # ('CN' in opt.category or 'HC' in opt.category )
                            data_health.append(OsJoin(root, feature, sub_fea))
                            path_diff = os.path.join(root, feature, sub_fea)
                            if 'diffusion' in sub_fea:
                                    path_fMRI = path_diff.replace('diffusion', 'rsfMRI').replace('Mean_Diffusion',
                                                                                                 'Mean_Function')
                            elif 'rsfMRI' in sub_fea:
                                    path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                                 'Mean_Diffusion')
                            data_health.append(path_fMRI)
                            if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                    label_health.append([1, 0, 0, 0])
                            if opt.n_classes == 3 and (opt.category == 'HC_MCI_SCD' or opt.category == 'HC_MCI_AD'):
                                label_health.append([1,0,0])
                            elif opt.n_classes == 2 and (opt.category == 'HC_MCI' or opt.category == 'HC_SCD' or opt.category == 'HC_AD'):
                                label_health.append([1,0])
                            HC_num = HC_num + 1
                        elif 'MCI' in sub_category and 'MCI' in opt.category:
                            data_MCI.append(OsJoin(root, feature, sub_fea))
                            path_diff = os.path.join(root, feature, sub_fea)
                            if 'diffusion' in sub_fea:
                                    path_fMRI = path_diff.replace('diffusion', 'rsfMRI').replace('Mean_Diffusion',
                                                                                                 'Mean_Function')
                            elif 'rsfMRI' in sub_fea:
                                    path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                                 'Mean_Diffusion')
                            data_MCI.append(path_fMRI)
                            if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                label_MCI.append([0, 0, 1, 0])
                            elif opt.n_classes == 3 and opt.category == 'HC_SCD_MCI':
                                label_MCI.append([0, 0, 1])
                            elif opt.n_classes == 3 and opt.category == 'HC_MCI_AD':
                                label_MCI.append([0, 1, 0])
                            elif opt.n_classes == 2 and ('SCD_MCI' in opt.category or 'HC_MCI' in opt.category ):
                                label_MCI.append([0, 1])
                            elif opt.n_classes == 2 and opt.category == 'MCI_AD':
                                label_MCI.append([1,0])
                            MCI_num = MCI_num + 1
                        elif 'SMC' in sub_category and ('SMC' in opt.category or 'SCD' in opt.category):
                            data_SCD.append(OsJoin(root, feature, sub_fea))
                            path_diff = os.path.join(root, feature, sub_fea)
                            if 'diffusion' in sub_fea:
                                    path_fMRI = path_diff.replace('diffusion', 'rsfMRI').replace('Mean_Diffusion',
                                                                                                 'Mean_Function')
                            elif 'rsfMRI' in sub_fea:
                                    path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                                 'Mean_Diffusion')
                            data_SCD.append(path_fMRI)
                            SCD_num = SCD_num + 1
                            if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                label_SCD.append([0,0,0,1])
                            if opt.n_classes == 3 and (opt.category == 'HC_SCD_MCI' or opt.category == 'HC_SCD_AD'):
                                label_SCD.append([0,1,0])
                            elif opt.n_classes == 2 and 'SCD_MCI' in opt.category:
                                label_SCD.append([1,0])
                            elif opt.n_classes == 2 and 'HC_SCD' in opt.category:
                                label_SCD.append([0,1])

                        elif 'AD' in sub_category and 'AD' in opt.category:
                            data_AD.append(OsJoin(root, feature, sub_fea))
                            if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                label_AD.append([0, 0, 0, 1])
                            elif opt.n_classes == 3 and opt.category == 'HC_MCI_AD':
                                label_AD.append([0, 0, 1])
                            elif opt.n_classes == 2 and ('MCI_AD' in opt.category or 'HC_AD' in opt.category):
                                label_AD.append([0, 1])
                            AD_num = AD_num + 1
                        # else:
                        #     print(sub_fea)
                elif 'OASIS-3' in sub_fea:  # data from OASIS
                    if str(sub) in str(sub_fea) and sub_fea not in OASIS_list:
                        OASIS_list.append(sub_fea)
                        sub_tmp = sub.split('_')[0]
                        # sub_index = list(sub_info_excel_OASIS ['OASISID']).index(sub)
                        # sub_category = sub_info_excel_OASIS ['dx1'][sub_index]
                        sessions_index =search_session(sub_tmp,list(sub_info_excel_OASIS ['OASISID']))
                        sessions_list = list(sub_info_excel_OASIS ['OASIS_session_label'])
                        index = search_near_session(sub_fea,sessions_index,sessions_list)
                        try:
                            sessions_name = sub_info_excel_OASIS ['dx1'][index]
                            sub_MMSE = sub_info_excel_OASIS ['MMSE'][index]
                            sub_category = sessions_name
                            # if 'Cognitively normal' in sub_category:
                                # CN = CN +1
                                # print(CN)
                                # print(sessions_index)
                                # print(index)
                                # print(sub_fea)
                        except:
                            print(index)
                        if sub_category is not None:
                            try :
                                if 'AD Dementia' in sub_category  and 'AD' in opt.category :
                                    # data_AD.append(OsJoin(root, feature, sub_fea))
                                    data_AD.append(OsJoin(root, feature, sub_fea))
                                    path_diff = os.path.join(root, feature, sub_fea)
                                    if 'diffusion' in sub_fea:
                                        path_fMRI = path_diff.replace('diffusion', 'rsfMRI').replace('Mean_Diffusion',
                                                                                                     'Mean_Function')
                                    elif 'rsfMRI' in sub_fea:
                                        path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                                     'Mean_Diffusion')
                                    data_AD.append(path_fMRI)
                                    if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                        label_AD.append([0, 0, 0, 1])
                                    elif opt.n_classes == 3 and opt.category == 'HC_MCI_AD':
                                        label_AD.append([0, 0, 1])
                                    elif opt.n_classes == 2 and ('MCI_AD' in opt.category or 'HC_AD' in opt.category):
                                        label_AD.append([0, 1])
                                    AD_num = AD_num + 1
                                elif 'Cognitively normal' in sub_category  and 'HC' in opt.category:
                                    data_health.append(OsJoin(root, feature, sub_fea))
                                    path_diff = os.path.join(root, feature, sub_fea)
                                    if 'diffusion' in sub_fea:
                                        path_fMRI = path_diff.replace('diffusion', 'rsfMRI').replace('Mean_Diffusion',
                                                                                                     'Mean_Function')
                                    elif 'rsfMRI' in sub_fea:
                                        path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                                     'Mean_Diffusion')
                                    data_health.append(path_fMRI)
                                    if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                        label_health.append([1, 0, 0, 0])
                                    elif opt.n_classes == 3 and (opt.category == 'HC_MCI_SCD' or opt.category == 'HC_MCI_AD'):
                                        label_health.append([1, 0, 0])
                                    elif opt.n_classes == 2 and (opt.category == 'HC_MCI' or opt.category == 'HC_SCD' \
                                                                 or opt.category == 'HC_AD'):
                                        label_health.append([1, 0])
                                    HC_num = HC_num + 1
                                else :#
                                       
                                            data_MCI.append(OsJoin(root, feature, sub_fea))
                                            path_diff = os.path.join(root, feature, sub_fea)
                                            if 'diffusion' in sub_fea:
                                                path_fMRI = path_diff.replace('diffusion', 'rsfMRI').replace('Mean_Diffusion',
                                                                                                             'Mean_Function')
                                            elif 'rsfMRI' in sub_fea:
                                                path_fMRI = path_diff.replace('rsfMRI', 'diffusion').replace('Mean_Function',
                                                                                                             'Mean_Diffusion')
                                            data_MCI.append(path_fMRI)
                                            MCI_num = MCI_num + 1
                                            if opt.n_classes == 4 and opt.category == 'HC_SCD_MCI_AD':
                                                label_MCI.append([0,0,1,0])
                                            elif opt.n_classes == 3 and opt.category == 'HC_SCD_MCI' :
                                                label_MCI.append([0,0,1])
                                            elif opt.n_classes == 3 and opt.category == 'HC_MCI_AD' :
                                                label_MCI.append([0,1,0])
                                            elif opt.n_classes == 2 and 'SCD_MCI' in opt.category:
                                                label_MCI.append([0,1])
                                            elif opt.n_classes == 2 and 'HC_MCI' in opt.category:
                                                label_MCI.append([0,1])
                            except:
                                print(sub_category)
                                print(opt.category)
                                continue

            else:
                continue
np.random.seed(opt.manual_seed)
if len(data_health) > 0:
    data_health = np.array(data_health).reshape(int(HC_num), 2)
    label_health = label_health[0:HC_num * fea_num:fea_num]
    # health_list = np.concatenate((data_health, np.array(label_health).reshape(int(HC_num / fea_num), 1)), axis=1)
    health_list = np.concatenate((data_health, np.array(label_health)), axis=1)
    np.random.shuffle(health_list)
    n_test_health = ceil(health_list.shape[0] * test_ratio)
    n_train_val_health = health_list.shape[0] - n_test_health
    train_val_list_health = health_list[0:n_train_val_health, :]
    test_list_health = health_list[n_train_val_health:health_list.shape[0], :]
if len(data_MCI) > 0:
    data_MCI=np.array(data_MCI).reshape(int(MCI_num), 2)
    label_MCI = label_MCI[0:MCI_num * fea_num:fea_num]
    MCI_list = np.concatenate((data_MCI, np.array(label_MCI)), axis=1)
    np.random.shuffle(MCI_list)
    n_test_MCI = ceil(MCI_list.shape[0] * test_ratio)
    n_train_val_MCI = MCI_list.shape[0] - n_test_MCI
    train_val_list_MCI = MCI_list[0:n_train_val_MCI, :]
    test_list_MCI = MCI_list[n_train_val_MCI:MCI_list.shape[0], :]
if len(data_SCD) > 0:
    data_SCD = np.array(data_SCD).reshape(int(SCD_num), 2)
    label_SCD = label_SCD[0:SCD_num * fea_num:fea_num]
    SCD_list = np.concatenate((data_SCD, np.array(label_SCD)), axis=1)
    np.random.shuffle(SCD_list)
    n_test_SCD = ceil(SCD_list.shape[0] * test_ratio)
    n_train_val_SCD = SCD_list.shape[0] - n_test_SCD
    train_val_list_SCD = SCD_list[0:n_train_val_SCD, :]
    test_list_SCD = SCD_list[n_train_val_SCD:SCD_list.shape[0], :]
kf = KFold(n_splits=300, shuffle=False)
n = 0
names = locals()
if len(data_health) > 0:
    for train_index, val_index in kf.split(train_val_list_health):
        n += 1
        names['train_fold%s_health'%n] = train_val_list_health[train_index]
        names['val_fold%s_health' % n] = train_val_list_health[val_index]
n = 0
if len(data_MCI) > 0:
    for train_index, val_index in kf.split(train_val_list_MCI):
        n += 1
        names['train_fold%s_MCI'%n] = train_val_list_MCI[train_index]
        names['val_fold%s_MCI' % n] = train_val_list_MCI[val_index]
n = 0
if len(data_SCD) > 0:
    for train_index, val_index in kf.split(train_val_list_SCD):
        n += 1
        names['train_fold%s_SCD'%n] = train_val_list_SCD[train_index]
        names['val_fold%s_SCD'%n] = train_val_list_SCD[val_index]
names2 = locals()
for i in range(1, n_fold+1):
    if len(data_health) > 0 and len(data_MCI) > 0 and len(data_SCD) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MCI'%i), names2.get('train_fold%s_SCD'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MCI'%i),  names2.get('val_fold%s_SCD'%i)))
        test_list = np.vstack((test_list_health, test_list_MCI, test_list_SCD))
    if len(data_health) > 0 and len(data_MCI) > 0 and len(data_SCD) == 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MCI'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MCI'%i)))
        test_list = np.vstack((test_list_health, test_list_MCI))
    if len(data_health) > 0 and len(data_MCI) == 0 and len(data_SCD) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_SCD'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_SCD'%i)))
        test_list = np.vstack((test_list_health, test_list_SCD))
    if len(data_health) == 0 and len(data_MCI) > 0 and len(data_SCD) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_MCI'%i), names2.get('train_fold%s_SCD'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_MCI'%i), names2.get('val_fold%s_SCD'%i)))
        test_list = np.vstack((test_list_MCI, test_list_SCD))
    np.random.seed(opt.manual_seed)
    np.random.shuffle(names2['train_list_fold%s'%i])
    np.random.shuffle(names2['val_list_fold%s'%i])

   # 按行堆叠
np.random.seed(opt.manual_seed)
np.random.shuffle(test_list)

csv_save_path = OsJoin(root, csv_save_dir)
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

for i in range(1, n_fold+1):
    with open(OsJoin(csv_save_path, 'train_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('train_list_fold%s'%i))
    with open(OsJoin(csv_save_path, 'val_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('val_list_fold%s'%i))


with open(OsJoin(csv_save_path, 'test.csv'), 'w', newline='') as f:  # 设置文件对象
    f_csv = csv.writer(f)
    f_csv.writerows(test_list)
