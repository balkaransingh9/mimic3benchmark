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
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}

e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}

def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}

def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.stay_id, 'Age': stays.age, 'Length of Stay': stays.los,
            'Mortality': stays.mortality}
    data.update(transform_gender(stays.gender))
    # data.update(transform_ethnicity(stays.ethnicity))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data = DataFrame(data).set_index('Icustay')
    data = data[['Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)

# List of diagnosis codes to include
diagnosis_labels = ['I169', 'I509', 'I2510', 'I4891', 'E119', 'N179', 'E785', 'J9690',
    'K219', 'N390', 'E7801', 'D649', 'E039', 'J189', 'E872', 'D62', 'J449', 'Z7901', 'R6520',
    'F329', 'A419', 'N189', 'J690', 'I129', 'F17200', 'I252', 'Z951', 'E871', 'I214', 'D696',
    'I348', 'Z87891', 'Z9861', 'Z794', 'I359', 'I120', 'R6521', 'J918', 'R001', 'G4733', 'J45998',
    'I9789', 'E875', 'E870', 'M109', 'I2789', 'J9819', 'I9581', 'I959', 'M810', 'N170', 'R569', 'N186',
    'I472', 'I428', 'I200', 'Z86718', 'F419', 'E1342', 'N400', 'E669', 'I2510', 'E876', 'I739', 'E860',
    'Z950', 'E861', 'N99821', 'I619', 'D631', 'F05', 'R7881', 'Y848', 'K922', 'R0902', 'Z66', 'Z853',
    'I5032', 'Y838', 'A0472', 'K7469', 'A419', 'B182', 'I5033', 'I469', 'J441', 'Z8546', 'F068',
    'L89159', 'D509', 'K7030', 'E6601', 'I4892', 'N99841', 'I209', 'F341', 'E46', 'I5022', 'E1140',
    'Z8673', 'I5023', 'D638', 'Y832', 'F1010', 'R197', 'R570', 'W19XXXA', 'R339', 'G40909', 'D500',
    'T814XXA', 'Z515', 'Y92199', 'R791', 'K766', 'G936', 'K567', 'E1129', 'K762', 'M1990', 'D689',
    'E873', 'K8592', 'Z7952', 'T827XXA', 'D72829', 'E11319', 'K5730', '4019', '4280', '41401',
    '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720', '2859', '2449', '486',
    '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070', '40390', '3051',
    '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867', '4241', '40391',
    '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749', '4168', '5180',
    '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111', 'V1251', '30000',
    '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652', '99811', '431',
    '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832', 'E8788', '00845',
    '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703', '2809', '5712',
    '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254', '42823', '28529',
    'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859', 'V667', 'E8497',
    '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770', 'V5865', '99662',
    '28860', '36201', '56210']

def extract_diagnosis_labels(diagnoses):
    """
    Transforms a diagnoses DataFrame into a wide-format label matrix of shape (stay_id x diagnosis_labels),
    with binary indicators and avoids pandas fragmentation by concatenating missing columns in bulk.
    """
    # Ensure a copy to avoid modifying original
    df = diagnoses.copy()
    df['value'] = 1
    # Pivot to get stays x icd_code matrix
    labels = (
        df[['stay_id', 'icd_code', 'value']]
          .drop_duplicates()
          .pivot(index='stay_id', columns='icd_code', values='value')
          .fillna(0)
          .astype(int)
    )

    # Identify any missing codes and add them in one step to avoid fragmentation
    missing = [code for code in diagnosis_labels if code not in labels.columns]
    if missing:
        zeros = pd.DataFrame(0, index=labels.index, columns=missing)
        labels = pd.concat([labels, zeros], axis=1)

    # Reorder columns as per diagnosis_labels list
    labels = labels.reindex(columns=diagnosis_labels)

    # Rename columns to 'Diagnosis <code>'
    rename_map = {code: f'Diagnosis {code}' for code in diagnosis_labels}
    labels = labels.rename(columns=rename_map)

    return labels

# Time series preprocessing
###################################

def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = pd.read_csv(fn).fillna('').astype(str)
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map = var_map[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']]
    var_map = var_map.rename({variable_column: 'variable', 'MIMIC LABEL': 'mimic_label'}, axis=1)
    var_map.columns = var_map.columns.str.lower()
    return var_map

def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='itemid', right_on='itemid')

def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
    to_rename[variable_column] = 'variable'
    var_ranges = dataframe_from_csv(fn, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='variable', keep='first')
    var_ranges.set_index('variable', inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]

def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = (events.variable == variable)
    v = events.loc[idx, 'value'].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, 'value'] = v
    return events

def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.str.contains('/')
    v.loc[idx] = v[idx].str.extract(r'^(\d+)/(\d+)$')[0]
    return v.astype(float)

def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.str.contains('/')
    v.loc[idx] = v[idx].str.extract(r'^(\d+)/(\d+)$')[1]
    return v.astype(float)

def clean_crr(df):
    v = Series(np.nan, index=df.index)
    df_str = df.value.astype(str)
    v.loc[df_str.isin(['Normal <3 secs', 'Brisk'])] = 0
    v.loc[df_str.isin(['Abnormal >3 secs', 'Delayed'])] = 1
    return v

def clean_fio2(df):
    v = df.value.astype(float).copy()
    is_str = df.value.apply(lambda x: isinstance(x, str)).values
    idx = df.valuenum.fillna('').str.lower().str.contains('torr') == False
    idx &= (is_str | (~is_str & (v > 1.0)))
    v.loc[idx] = v[idx] / 100.0
    return v

def clean_lab(df):
    v = df.value.copy()
    mask = v.apply(lambda x: isinstance(x, str) and not re.match(r'^(\d+(\.\d*)?|\.\d+)$', x))
    v.loc[mask] = np.nan
    return v.astype(float)

def clean_o2sat(df):
    v = df.value.copy()
    mask = v.apply(lambda x: isinstance(x, str) and not re.match(r'^(\d+(\.\d*)?|\.\d+)$', x))
    v.loc[mask] = np.nan
    v = v.astype(float)
    v.loc[v <= 1] = v[v <= 1] * 100.0
    return v

def clean_temperature(df):
    v = df.value.astype(float).copy()
    mask = df.valuenum.fillna('').str.lower().str.contains('f') | df.mimic_label.str.lower().str.contains('f') | (v >= 79)
    v.loc[mask] = (v[mask] - 32) * 5.0/9.0
    return v

def clean_weight(df):
    v = df.value.astype(float).copy()
    oz_mask = df.valuenum.fillna('').str.lower().str.contains('oz') | df.mimic_label.str.lower().str.contains('oz')
    v.loc[oz_mask] = v[oz_mask] / 16.0
    lb_mask = oz_mask | df.valuenum.fillna('').str.lower().str.contains('lb') | df.mimic_label.str.lower().str.contains('lb')
    v.loc[lb_mask] = v[lb_mask] * 0.453592
    return v

def clean_height(df):
    v = df.value.astype(float).copy()
    in_mask = df.valuenum.fillna('').str.lower().str.contains('in') | df.mimic_label.str.lower().str.contains('in')
    v.loc[in_mask] = np.round(v[in_mask] * 2.54)
    return v

# Map of cleaning functions
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

def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        mask = events.variable == var_name
        try:
            events.loc[mask, 'value'] = clean_fn(events[mask])
        except Exception as e:
            print(f"Error cleaning {var_name}: {e}")
            raise
    return events.loc[events.value.notnull()]