from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
import pandas as pd
from pandas import DataFrame, Series

from mimic3benchmark.util import dataframe_from_csv

###############################
# Non-time series preprocessing
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    return pd.DataFrame({'Gender': gender_series.fillna('').map(g_map).fillna(g_map['OTHER']).astype(int)})


e_map = {
    'ASIAN': 1, 'BLACK': 2, 'CARIBBEAN ISLAND': 2, 'HISPANIC': 3, 'SOUTH AMERICAN': 3, 
    'WHITE': 4, 'MIDDLE EASTERN': 4, 'PORTUGUESE': 4, 'AMERICAN INDIAN': 0, 'NATIVE HAWAIIAN': 0, 
    'UNABLE TO OBTAIN': 0, 'PATIENT DECLINED TO ANSWER': 0, 'UNKNOWN': 0, 'OTHER': 0, '': 0
}

def transform_ethnicity(ethnicity_series):
    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return pd.DataFrame({'Ethnicity': ethnicity_series.fillna('').map(e_map).fillna(e_map['OTHER']).astype(int)})


def assemble_episodic_data(stays, diagnoses):
    data = {
        'Icustay': stays.stay_id,
        'Age': stays.age,
        'Length of Stay': stays.los,
        'Mortality': stays.mortality,
        'Height': np.nan,
        'Weight': np.nan
    }
    data.update(transform_gender(stays.gender).to_dict(orient='series'))
    # data.update(transform_ethnicity(stays.ethnicity).to_dict(orient='series'))
    data = DataFrame(data).set_index('Icustay')
    data = data[['Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)


diagnosis_labels = [
    'I169', 'I509', 'I2510', 'I4891', 'E119', 'N179', 'E785', 'J9690', 'K219', 'N390', 'E7801', 'D649', 
    'E039', 'J189', 'E872', 'D62', 'J449', 'Z7901', 'R6520', 'F329', 'A419', 'N189', 'J690', 'I129', 
    'F17200', 'I252', 'Z951', 'E871', 'I214', 'D696', 'I348', 'Z87891', 'Z9861', 'Z794', 'I359', 'I120', 
    'R6521', 'J918', 'R001', 'G4733', 'J45998', 'I9789', 'E875', 'E870', 'M109', 'I2789', 'J9819', 
    'I9581', 'I959', 'M810', 'N170', 'R569', 'N186', 'I472', 'I428', 'I200', 'Z86718', 'F419', 'E1342', 
    'N400', 'E669', 'I2510', 'E876', 'I739', 'E860', 'Z950', 'E861', 'N99821', 'I619', 'D631', 'F05', 
    'R7881', 'Y848', 'K922', 'R0902', 'Z66', 'Z853', 'I5032', 'Y838', 'A0472', 'K7469', 'A419', 'B182', 
    'I5033', 'I469', 'J441', 'Z8546', 'F068', 'L89159', 'D509', 'K7030', 'E6601', 'I4892', 'N99841', 
    'I209', 'F341', 'E46', 'I5022', 'E1140', 'Z8673', 'I5023', 'D638', 'Y832', 'F1010', 'R197', 'R570', 
    'W19XXXA', 'R339', 'G40909', 'D500', 'T814XXA', 'Z515', 'Y92199', 'R791', 'K766', 'G936', 'K567', 
    'E1129', 'K762', 'M1990', 'D689', 'E873', 'K8592', 'Z7952', 'T827XXA', 'D72829', 'E11319', 'K5730'
]

def extract_diagnosis_labels(diagnoses):
    labels = diagnoses[['stay_id', 'icd_code']].drop_duplicates()
    labels['value'] = 1
    labels = labels.pivot(index='stay_id', columns='icd_code', values='value').fillna(0).astype(int)
    
    # Add missing columns efficiently
    missing_labels = [l for l in diagnosis_labels if l not in labels.columns]
    if missing_labels:
        labels = pd.concat([labels, pd.DataFrame(0, index=labels.index, columns=missing_labels)], axis=1)

    # Reorder columns to match original code behavior
    labels = labels[diagnosis_labels]

    # Rename for consistency
    labels.columns = ['Diagnosis ' + l for l in diagnosis_labels]

    return labels


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        try:
            idx = (events.variable == var_name)
            if idx.any():
                events.loc[idx, 'value'] = clean_fn(events.loc[idx])
        except Exception as e:
            print(f"Exception in clean_events for {var_name}: {e}")
            exit()
    return events.dropna(subset=['value'])


clean_fns = {
    'Capillary refill rate': clean_crr,
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    'Fraction inspired oxygen': clean_fio2,
    'Oxygen saturation': clean_o2sat,
    'Glucose': clean_lab,
    'pH': clean_lab,
    'Temperature': clean_temperature,
    'Weight': clean_weight,
    'Height': clean_height
}