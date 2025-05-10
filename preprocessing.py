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
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map.get(s, g_map['OTHER']))}

e_map = {
    'ASIAN': 1, 'BLACK': 2, 'CARIBBEAN ISLAND': 2,
    'HISPANIC': 3, 'SOUTH AMERICAN': 3,
    'WHITE': 4, 'MIDDLE EASTERN': 4, 'PORTUGUESE': 4,
    'AMERICAN INDIAN': 0, 'NATIVE HAWAIIAN': 0,
    'UNABLE TO OBTAIN': 0, 'PATIENT DECLINED TO ANSWER': 0,
    'UNKNOWN': 0, 'OTHER': 0, '': 0
}

def transform_ethnicity(ethnicity_series):
    def aggregate_ethnicity(s):
        return s.replace(' OR ', '/').split(' - ')[0].split('/')[0]
    series = ethnicity_series.fillna('').apply(aggregate_ethnicity)
    return {'Ethnicity': series.apply(lambda s: e_map.get(s, e_map['OTHER']))}

def assemble_episodic_data(stays, diagnoses):
    data = {
        'Icustay': stays.stay_id,
        'Age': stays.age,
        'Length of Stay': stays.los,
        'Mortality': stays.mortality
    }
    data.update(transform_gender(stays.gender))
    # data.update(transform_ethnicity(stays.ethnicity))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    df = DataFrame(data).set_index('Icustay')
    df = df[['Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    labels = extract_diagnosis_labels(diagnoses)
    return df.merge(labels, left_index=True, right_index=True)

# Diagnosis codes list (may include duplicates) - ensure uniqueness and order
_raw_diagnosis_labels = [
    'I169', 'I509', 'I2510', 'I4891', 'E119', 'N179', 'E785', 'J9690',
    'K219', 'N390', 'E7801', 'D649', 'E039', 'J189', 'E872', 'D62', 'J449',
    'Z7901', 'R6520', 'F329', 'A419', 'N189', 'J690', 'I129', 'F17200', 'I252',
    'Z951', 'E871', 'I214', 'D696', 'I348', 'Z87891', 'Z9861', 'Z794', 'I359',
    'I120', 'R6521', 'J918', 'R001', 'G4733', 'J45998', 'I9789', 'E875', 'E870',
    'M109', 'I2789', 'J9819', 'I9581', 'I959', 'M810', 'N170', 'R569', 'N186',
    'I472', 'I428', 'I200', 'Z86718', 'F419', 'E1342', 'N400', 'E669', 'E876',
    'I739', 'E860', 'Z950', 'E861', 'N99821', 'I619', 'D631', 'F05', 'R7881',
    'Y848', 'K922', 'R0902', 'Z66', 'Z853', 'I5032', 'Y838', 'A0472', 'K7469',
    'B182', 'I5033', 'I469', 'J441', 'Z8546', 'F068', 'L89159', 'D509', 'K7030',
    'E6601', 'I4892', 'N99841', 'I209', 'F341', 'E46', 'I5022', 'E1140', 'Z8673',
    'I5023', 'D638', 'Y832', 'F1010', 'R197', 'R570', 'W19XXXA', 'R339', 'G40909',
    'D500', 'T814XXA', 'Z515', 'Y92199', 'R791', 'K766', 'G936', 'K567', 'E1129',
    'K762', 'M1990', 'D689', 'E873', 'K8592', 'Z7952', 'T827XXA', 'D72829',
    'E11319', 'K5730', '4019', '4280', '41401', '42731', '25000', '5849', '2724',
    '51881', '53081', '5990', '2720', '2859', '2449', '486', '2762', '2851', '496',
    'V5861', '99592', '311', '0389', '5859', '5070', '40390', '3051', '412',
    'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867', '4241',
    '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760',
    '2749', '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856',
    '4271', '4254', '4111', 'V1251', '30000', '3572', '60000', '27800', '41400',
    '2768', '4439', '27651', 'V4501', '27652', '99811', '431', '28521', '2930',
    '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832', 'E8788', '00845',
    '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
    '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822',
    '25060', 'V1254', '42823', '28529', 'E8782', '30500', '78791', '78551',
    'E8889', '78820', '2800', '99859', 'V667', 'E8497', '79092', '5723', '3485',
    '5601', '25040', '570', '71590', '2869', '2763', '5770', 'V5865', '99662',
    '28860', '36201', '56210'
]
# Remove duplicates while preserving order
diagnosis_labels = list(dict.fromkeys(_raw_diagnosis_labels))

def extract_diagnosis_labels(diagnoses):
    """
    Transforms a diagnoses DataFrame into a stays x diagnosis label matrix without fragmentation.
    """
    df = diagnoses[['stay_id', 'icd_code']].drop_duplicates().copy()
    df['value'] = 1
    # Pivot to wide form
    labels = df.pivot(index='stay_id', columns='icd_code', values='value').fillna(0).astype(int)
    # Add missing codes in bulk
    missing = [c for c in diagnosis_labels if c not in labels.columns]
    if missing:
        zeros = pd.DataFrame(0, index=labels.index, columns=missing)
        labels = pd.concat([labels, zeros], axis=1)
    # Reorder and rename
    labels = labels.reindex(columns=diagnosis_labels)
    rename_map = {c: f'Diagnosis {c}' for c in diagnosis_labels}
    return labels.rename(columns=rename_map)

###################################
# Time series preprocessing
###################################

def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = pd.read_csv(fn).fillna('').astype(str)
    var_map = var_map[var_map.COUNT.astype(int) > 0]
    var_map = var_map[var_map[variable_column] != '']
    var_map = var_map[var_map.STATUS == 'ready']
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']]
    var_map = var_map.rename(columns={variable_column: 'variable', 'MIMIC LABEL': 'mimic_label'})
    var_map.columns = var_map.columns.str.lower()
    return var_map

def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, on='itemid')

def read_variable_ranges(fn, variable_column='LEVEL2'):
    cols = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    df = dataframe_from_csv(fn)
    df = df[cols].drop_duplicates(subset=variable_column)
    df = df.dropna()
    df = df.rename(columns={variable_column: 'variable',
                              'OUTLIER LOW': 'outlier_low',
                              'VALID LOW': 'valid_low',
                              'IMPUTE': 'impute',
                              'VALID HIGH': 'valid_high',
                              'OUTLIER HIGH': 'outlier_high'})
    return df.set_index('variable')

def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    mask = events.variable == variable
    v = events.loc[mask, 'value'].copy()
    rng = ranges.loc[variable]
    v[v < rng.outlier_low] = np.nan
    v[v > rng.outlier_high] = np.nan
    v[v < rng.valid_low] = rng.valid_low
    v[v > rng.valid_high] = rng.valid_high
    events.loc[mask, 'value'] = v
    return events

# Cleaning functions (sbp, dbp, crr, fio2, lab, o2sat, temperature, weight, height) remain unchanged
# ... [same clean_sbp, clean_dbp, clean_crr, clean_fio2, clean_lab, clean_o2sat,
#      clean_temperature, clean_weight, clean_height, clean_events] ...