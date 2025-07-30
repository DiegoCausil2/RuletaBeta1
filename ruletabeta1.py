# Ruleta Americana - Sistema de Predicción Avanzado con Machine Learning en Streamlit
import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import logging
import random
import warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import zscore
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ========== CONFIGURACIÓN INICIAL ==========
logging.basicConfig(level=logging.INFO)

# Filtrar advertencias específicas
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50%"
)

# ✅ Números válidos de la ruleta americana
ruleta_americana = [
    '0', '28', '9', '26', '30', '11', '7', '20', '32', '17', '5', '22', '34', '15', '3', '24',
    '36', '13', '1', '00', '27', '10', '25', '29', '12', '8', '19', '31', '18', '6', '21', '33',
    '16', '4', '23', '35', '14', '2'
]

# Propiedades predefinidas para cada número
number_properties = {}
reds = ['1', '3', '5', '7', '9', '12', '14', '16', '18', '19', '21', '23', '25', '27', '30', '32', '34', '36']
for num in ruleta_americana:
    # Color
    if num in ['0', '00']:
        color = 'green'
    else:
        color = 'red' if num in reds else 'black'
    
    # Paridad
    if num in ['0', '00']:
        parity = 'none'
    else:
        parity = 'even' if int(num) % 2 == 0 else 'odd'
    
    # Alto/Bajo
    if num in ['0', '00']:
        high_low = 'none'
    else:
        high_low = 'high' if int(num) >= 19 else 'low'
    
    # Sector
    if num in ['0', '00']:
        sector = 'zero'
    elif num in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
        sector = 'first'
    elif num in ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']:
        sector = 'second'
    else:
        sector = 'third'
    
    number_properties[num] = {
        'color': color,
        'parity': parity,
        'high_low': high_low,
        'sector': sector
    }

# ========== FUNCIONES AUXILIARES ==========
def parse_jugada(j):
    j = j.strip()
    return j if j in ruleta_americana else None

@st.cache_data(show_spinner=False, max_entries=3)
def calcular_estadisticas(jugadas_analisis):
    estadisticas = {
        n: {
            'frecuencia': 0,
            'ultima_aparicion': -1,
            'repeticiones': 0,
            'calor': 0,
            'score': 0,
            'z_score': 0
        } for n in ruleta_americana
    }

    if not jugadas_analisis:
        return estadisticas

    # Calcular frecuencias y última aparición
    for idx, numero in enumerate(jugadas_analisis):
        if numero not in ruleta_americana:
            continue
        estadisticas[numero]['frecuencia'] += 1
        estadisticas[numero]['ultima_aparicion'] = idx
        
        # Calcular nivel de "calor"
        distancia = len(jugadas_analisis) - idx - 1
        calor = max(0, 3 * np.exp(-0.1 * distancia))
        estadisticas[numero]['calor'] = max(estadisticas[numero]['calor'], calor)

    # Detectar repeticiones en últimas 20 jugadas
    ultimas_20 = jugadas_analisis[-20:] if len(jugadas_analisis) >= 20 else jugadas_analisis
    conteo_ultimas_20 = Counter(ultimas_20)
    for numero in ruleta_americana:
        count = conteo_ultimas_20.get(numero, 0)
        if count >= 3:
            estadisticas[numero]['repeticiones'] = 2.0
        elif count == 2:
            estadisticas[numero]['repeticiones'] = 1.5
        elif count == 1:
            estadisticas[numero]['repeticiones'] = 1.0

    # Calcular z-scores
    frecuencias = [estadisticas[n]['frecuencia'] for n in ruleta_americana]
    z_scores = zscore(frecuencias)
    for i, numero in enumerate(ruleta_americana):
        estadisticas[numero]['z_score'] = z_scores[i]
    
    # Calcular puntaje final
    for numero in ruleta_americana:
        datos = estadisticas[numero]
        datos['score'] = (
            datos['frecuencia'] * 0.4 +
            datos['calor'] * 0.3 +
            datos['repeticiones'] * 0.2 +
            datos['z_score'] * 0.1
        )
    
    return estadisticas

def obtener_vecinos(numero, radius=2):
    if numero not in ruleta_americana:
        return []
    i = ruleta_americana.index(numero)
    vecinos = []
    for offset in range(-radius, radius+1):
        if offset == 0:
            continue
        vecino = ruleta_americana[(i + offset) % len(ruleta_americana)]
        if vecino not in vecinos:
            vecinos.append(vecino)
    return vecinos

def obtener_terminacion(numero):
    if numero in ['0', '00']:
        return '0'
    return str(int(numero) % 10)

# Configuración de figuras y terminaciones
figuras_dict = {
    '1': ['1'],
    '2': ['2', '11', '20', '29'],
    '3': ['3', '12', '21', '30'],
    '4': ['4', '13', '22', '31'],
    '5': ['5', '14', '23', '32'],
    '6': ['6', '15', '24', '33'],
    '7': ['7', '16', '25', '34'],
    '8': ['8', '17', '26', '35'],
    '9': ['9', '18', '27', '36'],
    '10': ['10', '19', '28'],
    '11': ['11', '2', '29'],    
    '12': ['12', '3', '21'],
    '13': ['13', '4', '22', '31'],
    '14': ['14', '5', '23', '32'],
    '15': ['15', '6', '24', '33'],
    '16': ['16', '7', '25', '34'],
    '17': ['17', '8', '26', '35'],
    '18': ['18', '9', '27', '36'],
    '19': ['19', '10', '28'],
    '20': ['20', '2', '11', '29'],
    '21': ['21', '3', '12',],
    '22': ['22', '4', '13', '31'],
    '23': ['23', '5', '14', '32'],
    '24': ['24', '6', '15', '33'],
    '25': ['25', '7', '16', '34'],
    '26': ['26', '8', '17', '35'],
    '27': ['27', '9', '18', '36'],
    '28': ['28', '10', '19'],
    '29': ['29', '2', '11', '20'],
    '30': ['30', '3', '12'],
    '31': ['31', '4', '13', '22'],
    '32': ['32', '5', '14', '23'],
    '33': ['33', '6', '15', '24'],
    '34': ['34', '7', '16', '25'],
    '35': ['35', '8', '17', '26'],
    '36': ['36', '9', '18', '27'],
}

doble_figuras = { 
    '11': ['11', '22', '33', '00'],
    '22': ['22', '11', '33', '00'],
    '33': ['33', '11', '22', '00'],
    '00': ['00', '11', '22', '33']
}

terminaciones = {
    '0': ['0', '10', '20', '30'],
    '1': ['1', '11', '21', '31'],
    '2': ['2', '12', '22', '32'],
    '3': ['3', '13', '23', '33'],
    '4': ['4', '14', '24', '34'],
    '5': ['5', '15', '25', '35'],
    '6': ['6', '16', '26', '36'],
    '7': ['7', '17', '27'],
    '8': ['8', '18', '28'],
    '9': ['9', '19', '29']
}

pegados = { 
    '0':['1','2'],
    '00':['2','3'],
    '1': ['0','4'],
    '2': ['0','00','5'],
    '3': ['00','6'],
    '4': ['1','7'],
    '5': ['2','8'],
    '6': ['3','9'],
    '7': ['4','10'],
    '8': ['5','11'],
    '9': ['6','12'],
    '10': ['7','13'],
    '11': ['8','14'],
    '12': ['9','15'],
    '13': ['10','16'],
    '14': ['11','17'],
    '15': ['12','18'],
    '16': ['13','19'],
    '17': ['14','20'],
    '18': ['15','21'],
    '19': ['16','22'],
    '20': ['17','23'],
    '21': ['18','24'],
    '22': ['19','25'],
    '23': ['20','26'],
    '24': ['21','27'],
    '25': ['22','28'],
    '26': ['23','29'],
    '27': ['24','30'],
    '28': ['25','31'],
    '29': ['26','32'],
    '30': ['27','33'],
    '31': ['28','34'],
    '32': ['29','35'],
    '33': ['30','36'],
    '34': ['31'],
    '35': ['32'],
    '36': ['33'],
}

Puerta_Puerta = {
    '00': ['27','1'],
    '0':['28','2'],
    '1':['00','13'],
    '2':['0','14'],
    '3':['24','15'],
    '4':['23','16'],
    '5':['22','17'],
    '6':['21','18'],
    '7':['20','11'],
    '8':['19','12'],
    '9':['26','28'],
    '10':['25','27'],
    '11':['7','30'],
    '12':['8','29'],
    '13':['1','36'],
    '14':['2','35'],
    '15':['3','34'],
    '16':['4','33'],
    '17':['5','32'],
    '18':['6','31'],
    '19':['31','8'],
    '20':['32','7'],
    '21':['33','6'],
    '22':['34','5'],
    '23':['35','4'],
    '24':['36','3'],
    '25':['29','10'],
    '26':['30','9'],
    '27':['10','00'],
    '28':['9','0'],
    '29':['12','25'],
    '30':['11','26'],
    '31':['18','19'],
    '32':['17','20'],
    '33':['16','21'],
    '34':['15','22'],
    '35':['14','23'],
    '36':['13','24']
                 
}

numeros_con_espejo = {
    '0': ['00', '0', '6', '9', '12', '21', '13', '31', '16', '19', '23', '32', '26', '29'],
    '00': ['0', '00', '6', '9', '12', '21', '13', '31', '16', '19', '23', '32', '26', '29'],
    '6': ['9', '6', '0', '00', '12', '21', '13', '31', '16', '19', '23', '32', '26', '29'],
    '9': ['6', '9', '0', '00', '12', '21', '13', '31', '16', '19', '23', '32', '26', '29'],
    '12': ['21', '12', '0', '00', '6', '9', '13', '31', '16', '19', '23', '32', '26', '29'],
    '21': ['12', '21', '0', '00', '6', '9', '13', '31', '16', '19', '23', '32', '26', '29'],
    '13': ['31', '13', '0', '00', '6', '9', '12', '21', '16', '19', '23', '32', '26', '29'],
    '31': ['13', '31', '0', '00', '6', '9', '12', '21', '16', '19', '23', '32', '26', '29'],
    '16': ['19', '16', '0', '00', '6', '9', '12', '21', '13', '31', '23', '32', '26', '29'],
    '19': ['16', '19', '0', '00', '6', '9', '12', '21', '13', '31', '23', '32', '26', '29'],
    '23': ['32', '23', '0', '00', '6', '9', '12', '21', '13', '31', '16', '19', '26', '29'],
    '32': ['23', '32', '0', '00', '6', '9', '12', '21', '13', '31', '16', '19', '26', '29'],
    '26': ['29', '26', '0', '00', '6', '9', '12', '21', '13', '31', '16', '19', '23', '32'],
    '29': ['26', '29', '0', '00', '6', '9', '12', '21', '13', '31', '16', '19', '23', '32']
}

arañita = {
    '1':['1','3','5','7','9','14','19','21','23','25','27'],
    '3':['1','3','5','7','9','14','19','21','23','25','27'],
    '5':['1','3','5','7','9','14','19','21','23','25','27'],
    '7':['1','3','5','7','9','14','19','21','23','25','27'],
    '9':['1','3','5','7','9','14','19','21','23','25','27'],
    '14':['1','3','5','7','9','14','19','21','23','25','27'],
    '19':['1','3','5','7','9','14','19','21','23','25','27'],
    '21':['1','3','5','7','9','14','19','21','23','25','27'],
    '23':['1','3','5','7','9','14','19','21','23','25','27'],
    '25':['1','3','5','7','9','14','19','21','23','25','27'],
    '27':['1','3','5','7','9','14','19','21','23','25','27'],
}

rojo_con_rojo_izq = {
    '12': ['21', '16', '19', '18', '17', '14', '25', '27'],
    '14': ['23', '16', '21', '18', '19', '12', '25', '27'],
    '16': ['25', '14', '21', '18', '19', '12', '23', '27'],
    '18': ['27', '16', '21', '14', '19', '12', '23', '25'],
    '19': ['12', '14', '16', '18', '21', '23', '25', '27'],
    '21': ['14', '16', '18', '19', '12', '23', '25', '27'],
    '23': ['16', '18', '19', '21', '12', '14', '25', '27'],
    '25': ['18', '19', '21', '23', '12', '14', '16', '27'],
    '27': ['19', '21', '23', '25', '12', '14', '16', '18']
}

Negro_con_negro_izq = {
    '2' : ['35', '4', '33', '6', '31', '8', '29', '10'],
    '4' : ['2', '35', '33', '6', '31', '8', '29', '10'],
    '6' : ['4', '2', '35', '33', '31', '8', '29', '10'],
    '8' : ['6', '4', '2', '35', '33', '31', '29', '10'],
    '10': ['8', '6', '4', '2', '35', '33', '31', '29'],
    '29': ['10', '8', '6', '4', '2', '35', '33', '31'],
    '31': ['29', '10', '8', '6', '4', '2', '35', '33'],
    '33': ['31', '29', '10', '8', '6', '4', '2', '35'],
    '35': ['33', '31', '29', '10', '8', '6', '4', '2']
}

Rojo_con_rojo_der = {
    '1' : ['36', '3', '34', '5', '32', '7', '30', '9'],
    '3' : ['1', '36', '34', '5', '32', '7', '30', '9'],
    '5' : ['1', '36', '3', '34', '32', '7', '30', '9'],
    '7' : ['1', '36', '3', '34', '5', '32', '30', '9'],
    '9' : ['1', '36', '3', '34', '5', '32', '7', '30'],
    '30': ['1', '36', '3', '34', '5', '32', '7', '9'],
    '32': ['1', '36', '3', '34', '5', '7', '30', '9'],
    '34': ['1', '36', '3', '5', '32', '7', '30', '9'],
    '36': ['1', '3', '5', '7', '9', '32', '34', '30']
}

Negro_con_negro_der = { 
    '11' :['13', '24', '15', '22', '17', '20', '26', '28'],
    '13' :['24', '15', '22', '17', '20', '11', '26', '28'],
    '15' :['13', '24', '22', '17', '20', '11', '26', '28'],
    '17' :['13', '24', '15', '22', '20', '11', '26', '28'],
    '20' :['13', '24', '15', '22', '17', '11', '26', '28'],
    '22' :['13', '24', '15', '17', '20', '11', '26', '28'],
    '24' :['13', '15', '22', '17', '20', '11', '26', '28'],
    '26' :['13', '24', '15', '22', '17', '20', '11', '28'],
    '28' :['13', '24', '15', '22', '17', '20', '11', '26'],
}

pequeños_rojos = {
    '1' :['3', '5', '7', '9'],
    '3' :['1', '5', '7', '9'],
    '5' :['1', '3', '7', '9'],
    '7' :['1', '3', '5', '9'],
    '9' :['1', '3', '5', '7']
}

pequeño_negros = {
    '2' :['4', '6', '8', '10'],
    '4' :['2', '6', '8', '10'],
    '6' :['2', '4', '8', '10'],
    '8' :['2', '4', '6', '10'],
    '10':['2', '4', '6', '8']
}

veinte_rojos = {
    '21' :['23', '25', '27'],
    '23' :['21', '25', '27'],
    '25' :['21', '23', '25'],
    '27' :['21', '23', '25']
}

veinte_negros = {
    '20' :['22', '24', '26', '28', '29'],
    '22' :['20', '24', '26', '28', '29'],
    '24' :['20', '22', '26', '28', '29'],
    '26' :['20', '22', '24', '28', '29'],
    '28' :['20', '22', '24', '26', '29'],
    '29' :['20', '22', '24', '26', '28']                                                         
}

treinta_rojo = {
    '30' : ['32', '34', '36'],
    '32' : ['30', '34', '36'],
    '34' : ['30', '32', '36'],
    '36' : ['30', '32', '34']
}

treinta_negro = {
    '31' :['33', '35'],
    '33' :['31', '35'],
    '35' :['31', '33']
}

docena_1 = { 
    '1' :['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    '2' :['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    '3' :['1', '2', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    '4' :['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '12'],
    '5' :['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12'],
    '6' :['1', '2', '3', '4', '5', '7', '8', '9', '10', '11', '12'],
    '7' :['1', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12'],
    '8' :['1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12'],
    '9' :['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12'],
    '10' :['1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12'],
    '11' :['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12'],
    '12' :['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
}

docena_2 = { 
    '13' : ['14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'],
    '14' : ['13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'],
    '15' : ['13', '14', '16', '17', '18', '19', '20', '21', '22', '23', '24'],
    '16' : ['13', '14', '15', '17', '18', '19', '20', '21', '22', '23', '24'],
    '17' : ['13', '14', '15', '16', '18', '19', '20', '21', '22', '23', '24'],
    '18' : ['13', '14', '15', '16', '17', '19', '20', '21', '22', '23', '24'],
    '19' : ['13', '14', '15', '16', '17', '18', '20', '21', '22', '23', '24'],
    '20' : ['13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24'],
    '21' : ['13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24'],
    '22' : ['13', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24'],
    '23' : ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24'],
    '24' : ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
}

docena_3 = {
    '25': ['26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36'],
    '26': ['25', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36'],
    '27': ['25', '26', '28', '29', '30', '31', '32', '33', '34', '35', '36'],
    '28': ['25', '26', '27', '29', '30', '31', '32', '33', '34', '35', '36'],
    '29': ['25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '36'],
    '30': ['25', '26', '27', '28', '29', '31', '32', '33', '34', '35', '36'],
    '31': ['25', '26', '27', '28', '29', '30', '32', '33', '34', '35', '36'],
    '32': ['25', '26', '27', '28', '29', '30', '31', '33', '34', '35', '36'],
    '33': ['25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36'],
    '34': ['25', '26', '27', '28', '29', '30', '31', '32', '33', '35', '36'],
    '35': ['25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '36'],
    '36': ['25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
}

columna_1 = {
    '3': ['6', '9', '12', '15', '18', '21', '24', '27', '30', '33', '36'],
    '6': ['3', '9', '12', '15', '18', '21', '24', '27', '30', '33', '36'],
    '9': ['3', '6', '12', '15', '18', '21', '24', '27', '30', '33', '36'],
    '12': ['3', '6', '9', '15', '18', '21', '24', '27', '30', '33', '36'],
    '15': ['3', '6', '9', '12', '18', '21', '24', '27', '30', '33', '36'],
    '18': ['3', '6', '9', '12', '15', '21', '24', '27', '30', '33', '36'],
    '21': ['3', '6', '9', '12', '15', '18', '24', '27', '30', '33', '36'],
    '24': ['3', '6', '9', '12', '15', '18', '21', '27', '30', '33', '36'],
    '27': ['3', '6', '9', '12', '15', '18', '21', '24', '30', '33', '36'],
    '30': ['3', '6', '9', '12', '15', '18', '21', '24', '27', '33', '36'],
    '33': ['3', '6', '9', '12', '15', '18', '21', '24', '27', '30', '36'],
    '36': ['3', '6', '9', '12', '15', '18', '21', '24', '27', '30', '33']
}

columna_2 ={
    '2': ['5', '8', '11', '14', '17', '20', '23', '26', '29', '32', '35'],
    '5': ['2', '8', '11', '14', '17', '20', '23', '26', '29', '32', '35'],
    '8': ['2', '5', '11', '14', '17', '20', '23', '26', '29', '32', '35'],
    '11':['2', '5', '8', '14', '17', '20', '23', '26', '29', '32', '35'],
    '14':['2', '5', '8', '11', '17', '20', '23', '26', '29', '32', '35'],
    '17':['2', '5', '8', '11', '14', '20', '23', '26', '29', '32', '35'],
    '20':['2', '5', '8', '11', '14', '17', '23', '26', '29', '32', '35'],
    '23':['2', '5', '8', '11', '14', '17', '20', '26', '29', '32', '35'],
    '26':['2', '5', '8', '11', '14', '17', '20', '23', '29', '32', '35'],
    '29':['2', '5', '8', '11', '14', '17', '20', '23', '26', '32', '35'],
    '32':['2', '5', '8', '11', '14', '17', '20', '23', '26', '29', '35'],
    '35':['2', '5', '8', '11', '14', '17', '20', '23', '26', '29', '32']
}

columna_3 = {
    '1':['4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34'],
    '4':['1', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34'],
    '7':['1', '4', '10', '13', '16', '19', '22', '25', '28', '31', '34'],
    '10':['1', '4', '7', '13', '16', '19', '22', '25', '28', '31', '34'],
    '13':['1', '4', '7', '10', '16', '19', '22', '25', '28', '31', '34'],
    '16':['1', '4', '7', '10', '13', '19', '22', '25', '28', '31', '34'],
    '19':['1', '4', '7', '10', '13', '16', '22', '25', '28', '31', '34'],
    '22':['1', '4', '7', '10', '13', '16', '19', '25', '28', '31', '34'],
    '25':['1', '4', '7', '10', '13', '16', '19', '22', '28', '31', '34'],
    '28':['1', '4', '7', '10', '13', '16', '19', '22', '25', '31', '34'],
    '31':['1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '34'],
    '34':['1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31']
}

def obtener_espejo(n):
    espejo_map = {
        '0': '00',
        '00': '0',
        '6': '9', '9': '6',
        '12': '21', '21': '12',
        '13': '31', '31': '13',
        '16': '19', '19': '16',
        '23': '32', '32': '23',
        '26': '29', '29': '26'  
    }
    return espejo_map.get(n, None)

# Lista completa de patrones para usar en predicciones
TODOS_PATRONES = [
    Puerta_Puerta, arañita, rojo_con_rojo_izq, Negro_con_negro_izq, 
    Rojo_con_rojo_der, Negro_con_negro_der, pequeños_rojos, pequeño_negros,
    veinte_rojos, veinte_negros, treinta_rojo, treinta_negro, docena_1, 
    docena_2, docena_3, columna_1, columna_2, columna_3, numeros_con_espejo
]

@st.cache_data(show_spinner=False, max_entries=3)
def aplicar_patrones_experiencia(jugadas):
    patrones_detectados = []
    if len(jugadas) < 4:
        return patrones_detectados

    # Patrones de espejos diferidos
    for i in range(len(jugadas)-3):
        espejo = obtener_espejo(jugadas[i])
        if espejo and espejo in jugadas[i+1:i+4]:
            patrones_detectados.append(("Espejo diferido", jugadas[i], espejo))

    # Patrones de figuras de cierre (A-B-C-A)
    for i in range(len(jugadas)-3):
        if jugadas[i] == jugadas[i+3] and jugadas[i+1] != jugadas[i+2]:
            patrones_detectados.append(("Figura de cierre", ', '.join(jugadas[i:i+4])))
    
    # Patrones de sumas (evitando 0 y 00)
    for i in range(len(jugadas)-2):
        if jugadas[i] in ['0', '00'] or jugadas[i+1] in ['0', '00']:
            continue  # Saltar números no sumables
            
        try:
            n1 = int(jugadas[i])
            n2 = int(jugadas[i+1])
            suma = n1 + n2
            if 1 <= suma <= 36:
                suma_str = str(suma)
                if suma_str in jugadas[i+2:i+5]:
                    patrones_detectados.append(("Suma simple", f"{n1}+{n2}={suma_str}"))
        except:
            continue
    
    # Patrones de doble figuras
    ultimas_15 = jugadas[-15:] if len(jugadas) >= 15 else jugadas
    for num in set(ultimas_15):
        if num in doble_figuras:
            for n in doble_figuras[num]:
                if n in ultimas_15 and n != num:
                    patrones_detectados.append(("Doble figura", f"{num} y {n}"))
    
    # Patrones de secuencias (3+ números consecutivos)
    for i in range(len(jugadas)-2):
        try:
            nums = [int(jugadas[i]), int(jugadas[i+1]), int(jugadas[i+2])]
            if abs(nums[0] - nums[1]) == 1 and abs(nums[1] - nums[2]) == 1:
                patrones_detectados.append(("Secuencia", f"{jugadas[i]}-{jugadas[i+1]}-{jugadas[i+2]}"))
        except:
            continue
    
    return patrones_detectados

def calcular_score_dinamico_usuario():
    total_jugadas = len(st.session_state.jugadas_usuario)
    for numero, data in st.session_state.estadisticas_usuario.items():
        if data['ultima_aparicion'] >= 0:
            tiempo_fuera = total_jugadas - data['ultima_aparicion'] - 1
        else:
            tiempo_fuera = total_jugadas
        
        penalizacion_frio = 3 if tiempo_fuera > 100 else 0
            
        freq_weight = 0.4
        heat_weight = 0.3
        rep_weight = 0.2
        z_weight = 0.1
        
        if data['z_score'] > 1.5:
            z_weight = 0.2
            freq_weight = 0.3
        
        data['score'] = (
            data['frecuencia'] * freq_weight +
            data['calor'] * heat_weight +
            data['repeticiones'] * rep_weight +
            data['z_score'] * z_weight - 
            penalizacion_frio
        )

def actualizar_estadisticas_usuario(jugada_actual):
    total_jugadas = len(st.session_state.jugadas_usuario)
    
    # Actualizar estadísticas
    data = st.session_state.estadisticas_usuario[jugada_actual]
    data['frecuencia'] += 1
    data['ultima_aparicion'] = total_jugadas - 1
    
    # Actualizar calor
    for numero in ruleta_americana:
        data_num = st.session_state.estadisticas_usuario[numero]
        if data_num['ultima_aparicion'] >= 0:
            tiempo_fuera = total_jugadas - data_num['ultima_aparicion'] - 1
        else:
            tiempo_fuera = total_jugadas
        
        if tiempo_fuera == 0:
            data_num['calor'] = 3.0
        else:
            data_num['calor'] = max(0, 3 * np.exp(-0.1 * tiempo_fuera))
    
    # Recalcular repeticiones
    ultimas_20 = st.session_state.jugadas_usuario[-20:] if len(st.session_state.jugadas_usuario) >= 20 else st.session_state.jugadas_usuario
    conteo_ultimas_20 = Counter(ultimas_20)
    for numero in ruleta_americana:
        count = conteo_ultimas_20.get(numero, 0)
        if count >= 3:
            st.session_state.estadisticas_usuario[numero]['repeticiones'] = 2.0
        elif count == 2:
            st.session_state.estadisticas_usuario[numero]['repeticiones'] = 1.5
        elif count == 1:
            st.session_state.estadisticas_usuario[numero]['repeticiones'] = 1.0
        else:
            st.session_state.estadisticas_usuario[numero]['repeticiones'] = 0
    
    # Actualizar z-scores
    frecuencias = [st.session_state.estadisticas_usuario[n]['frecuencia'] for n in ruleta_americana]
    z_scores = zscore(frecuencias)
    for i, numero in enumerate(ruleta_americana):
        st.session_state.estadisticas_usuario[numero]['z_score'] = z_scores[i]
    
    # Actualizar puntajes
    calcular_score_dinamico_usuario()

def extraer_caracteristicas(jugadas):
    """Extrae características para el modelo de ML"""
    features = []
    
    # 1. Frecuencias básicas
    freq = Counter(jugadas)
    freq_features = [freq.get(n, 0) for n in ruleta_americana]
    
    # 2. Frecuencias ponderadas
    weighted_freq = {n: 0 for n in ruleta_americana}
    decay = 0.9
    for i, num in enumerate(reversed(jugadas)):
        weighted_freq[num] += decay ** i
    weighted_features = [weighted_freq[n] for n in ruleta_americana]
    
    # 3. Características de la última jugada
    last_num = jugadas[-1] if jugadas else '0'
    last_props = number_properties[last_num]
    
    # 4. Características de las últimas 3 jugadas
    last_3 = jugadas[-3:] if len(jugadas) >= 3 else jugadas
    colors = [number_properties[n]['color'] for n in last_3 if n in number_properties]
    parities = [number_properties[n]['parity'] for n in last_3 if n in number_properties]
    sectors = [number_properties[n]['sector'] for n in last_3 if n in number_properties]
    
    # 5. Patrones detectados
    patrones = aplicar_patrones_experiencia(jugadas)
    pattern_count = len(patrones)
    
    # Combinar características
    features = (
        freq_features + 
        weighted_features +
        [
            1 if last_props['color'] == 'red' else 0,
            1 if last_props['color'] == 'black' else 0,
            1 if last_props['color'] == 'green' else 0,
            1 if last_props['parity'] == 'even' else 0,
            1 if last_props['parity'] == 'odd' else 0,
            1 if last_props['high_low'] == 'high' else 0,
            1 if last_props['high_low'] == 'low' else 0,
            pattern_count
        ]
    )
    
    return np.array(features)

@st.cache_data(show_spinner=False, max_entries=3)
def preparar_datos_ml(jugadas):
    if len(jugadas) < 11:
        return np.array([]), np.array([])
    
    ventanas = [jugadas[i-10:i] for i in range(10, len(jugadas))]
    y = jugadas[10:]
    
    X = Parallel(n_jobs=-1)(delayed(extraer_caracteristicas)(ventana) for ventana in ventanas)
    
    return np.array(X), np.array(y)

# 🔹 Evita reconstruir modelo si no cambian las jugadas
@st.cache_resource(show_spinner=False)
def entrenar_modelo(jugadas):
    X, y = preparar_datos_ml(jugadas)
    if len(X) == 0:
        return None, None, None, None, None
        
    test_size = max(2, int(0.1 * len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_components = min(30, X_train_scaled.shape[1] // 2)
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    model = SGDClassifier(
        loss='log_loss',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    try:
        start_time = time.time()
        model.fit(X_train_pca, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_pca)
        st.session_state.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        initial_type = [('float_input', FloatTensorType([None, X_train_pca.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        onnx_bytes = onnx_model.SerializeToString()
        onnx_session = rt.InferenceSession(onnx_bytes)
        
        st.session_state.modelo_actualizado = True
        return model, pca, scaler, onnx_session, onnx_bytes
    
    except Exception as e:
        logging.error(f"Error en entrenamiento: {str(e)}")
        st.error(f"Error en entrenamiento: {str(e)}")
        return None, None, None, None, None

def actualizar_modelo_parcial(X_new, y_new):
    """Actualiza el modelo con nuevos datos en batch"""
    if (st.session_state.modelo is None or 
        st.session_state.scaler is None or 
        st.session_state.pca is None):
        return
    
    BATCH_SIZE = 10
    st.session_state.X_batch.append(X_new)
    st.session_state.y_batch.append(y_new)
    
    if len(st.session_state.X_batch) >= BATCH_SIZE:
        try:
            X_batch_scaled = st.session_state.scaler.transform(st.session_state.X_batch)
            X_batch_pca = st.session_state.pca.transform(X_batch_scaled)
            
            # Convertir y_batch a array de numpy
            y_batch_array = np.array(st.session_state.y_batch)
            
            # Si es la primera actualización parcial, pasar todas las clases
            if not hasattr(st.session_state.modelo, 'classes_'):
                clases = np.array(ruleta_americana)
                st.session_state.modelo.partial_fit(X_batch_pca, y_batch_array, classes=clases)
            else:
                st.session_state.modelo.partial_fit(X_batch_pca, y_batch_array)
            
            st.session_state.modelo_actualizado = True
            
            st.session_state.X_batch = []
            st.session_state.y_batch = []
            
            if st.session_state.modelo_actualizado:
                initial_type = [('float_input', FloatTensorType([None, X_batch_pca.shape[1]]))]
                onnx_model = convert_sklearn(st.session_state.modelo, initial_types=initial_type)
                
                st.session_state.onnx_bytes = onnx_model.SerializeToString()
                st.session_state.onnx_session = rt.InferenceSession(st.session_state.onnx_bytes)
                st.session_state.modelo_actualizado = False
        except Exception as e:
            logging.error(f"Error en actualización parcial: {str(e)}", exc_info=True)
            st.error(f"Error en actualización parcial: {str(e)}")

def predecir_numeros():
    # Solo usar jugadas manuales del usuario
    if not st.session_state.jugadas_usuario or len(st.session_state.jugadas_usuario) < 10:
        return [], [], []
    
    if st.session_state.modelo is None or st.session_state.onnx_session is None:
        return [], [], []

    # Preparar datos para predicción (últimas 10 jugadas MANUALES)
    ultimas_jugadas = st.session_state.jugadas_usuario[-10:]
    X = extraer_caracteristicas(ultimas_jugadas)
    
    # Preprocesamiento
    try:
        X_scaled = st.session_state.scaler.transform([X])
        X_pca = st.session_state.pca.transform(X_scaled)
        
        # Inferencia con ONNX
        input_name = st.session_state.onnx_session.get_inputs()[0].name
        label_name = st.session_state.onnx_session.get_outputs()[0].name
        probabilidades = st.session_state.onnx_session.run([label_name], {input_name: X_pca.astype(np.float32)})[0][0]
        
        clases = ruleta_americana
        prob_clases = list(zip(clases, probabilidades))
        prob_clases.sort(key=lambda x: x[1], reverse=True)
        
        # Obtener top 18 predicciones
        top_18 = [n for n, _ in prob_clases[:18]]
        
        # Aplicar patrones de experiencia
        patrones = aplicar_patrones_experiencia(st.session_state.jugadas_usuario)
        sugeridos_extras = set()
        
        for patron in patrones:
            elementos = patron[1:]
            for elem in elementos:
                if isinstance(elem, str) and elem in ruleta_americana:
                    sugeridos_extras.add(elem)
        
        # Añadir números relacionados (vecinos, terminaciones, patrones)
        for n in top_18:
            # Vecinos
            vecinos = obtener_vecinos(n, radius=1)
            for v in vecinos:
                if v in ruleta_americana:
                    sugeridos_extras.add(v)
            
            # Terminaciones
            term = obtener_terminacion(n)
            if term in terminaciones:
                for t in terminaciones[term]:
                    sugeridos_extras.add(t)
            
            # Todos los patrones adicionales
            for pat_dict in TODOS_PATRONES:
                if n in pat_dict:
                    for num_pat in pat_dict[n]:
                        if num_pat in ruleta_americana:
                            sugeridos_extras.add(num_pat)
        
        # Añadir sugerencias extras
        for n in sugeridos_extras:
            if n not in top_18 and n in ruleta_americana:
                top_18.append(n)
        
        # Garantizar 18 elementos con números aleatorios únicos
        while len(top_18) < 18:
            rnd = random.choice(ruleta_americana)
            if rnd not in top_18:
                top_18.append(rnd)
        
        # Tomar solo los primeros 18 elementos
        todos_sugeridos = top_18[:18]
        
        # Dividir en dos grupos de 9
        calientes = todos_sugeridos[:9]
        estrategicos = todos_sugeridos[9:18]
        
        # Guardar predicción
        st.session_state.prediction_history.append({
            'calientes': calientes,
            'estrategicos': estrategicos,
            'timestamp': time.time()
        })
        
        return calientes, estrategicos, patrones
    
    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        st.warning("Error al generar predicciones")
        return [], [], []

def mostrar_metricas_texto():
    if not st.session_state.performance_metrics:
        return
    
    metrics = st.session_state.performance_metrics
    st.markdown("**Métricas del Modelo:**")
    st.markdown(f"- Precisión: {metrics['precision']:.2%}")
    st.markdown(f"- Exactitud: {metrics['accuracy']:.2%}")
    st.markdown(f"- Recall: {metrics['recall']:.2%}")
    st.markdown(f"- F1-Score: {metrics['f1']:.2%}")

def mostrar_historial_simple():
    if not st.session_state.prediction_history:
        return
    
    history = st.session_state.prediction_history[-5:]
    st.markdown("**Historial Reciente:**")
    
    for i, pred in enumerate(reversed(history)):
        hora = time.strftime('%H:%M:%S', time.localtime(pred['timestamp']))
        st.markdown(f"**{hora}**")
        st.markdown(f"🔥: {', '.join(pred['calientes'])}")
        st.markdown(f"🎯: {', '.join(pred['estrategicos'])}")
        st.markdown("---")

def inicializar_estadisticas():
    return {
        n: {
            'frecuencia': 0,
            'ultima_aparicion': -1,
            'repeticiones': 0,
            'calor': 0,
            'score': 0,
            'z_score': 0
        } for n in ruleta_americana
    }

def cargar_jugadas_desde_texto(texto):
    jugadas_match = re.findall(r'\b(00|0|[1-9]|[1-2][0-9]|3[0-6])\b', texto)
    return [j for j in jugadas_match if j in ruleta_americana]

def actualizar_texto_jugadas(nueva_jugada):
    """Actualiza el área de texto con la nueva jugada al inicio"""
    current_text = st.session_state.jugadas_texto_data
    if current_text:
        jugadas_lista = [j.strip() for j in current_text.split(',') if j.strip()]
    else:
        jugadas_lista = []
    
    # Insertar nueva jugada al principio
    jugadas_lista.insert(0, nueva_jugada)
    
    # Limitar a 500 jugadas
    if len(jugadas_lista) > 500:
        jugadas_lista = jugadas_lista[:500]
    
    st.session_state.jugadas_texto_data = ', '.join(jugadas_lista)

def agregar_jugada_manual(nueva_jugada):
    """Agrega una jugada manteniendo el límite de 500"""
    if len(st.session_state.jugadas_usuario) < 500:
        st.session_state.jugadas_usuario.append(nueva_jugada)
    else:
        # Mantener solo las últimas 500 jugadas
        st.session_state.jugadas_usuario = st.session_state.jugadas_usuario[1:] + [nueva_jugada]

def inicializar_session_state():
    keys = [
        'base_historica', 'jugadas_usuario', 'modelo', 'fase_inicial', 'estadisticas_usuario',
        'performance_metrics', 'prediction_history', 'pca', 'scaler',
        'onnx_session', 'modelo_actualizado', 'X_batch', 'y_batch', 'onnx_bytes',
        'jugadas_texto_data'
    ]
    
    defaults = {
        'base_historica': [],
        'jugadas_usuario': [],
        'modelo': None,
        'fase_inicial': True,
        'estadisticas_usuario': inicializar_estadisticas(),
        'performance_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
        'prediction_history': [],
        'pca': None,
        'scaler': None,
        'onnx_session': None,
        'modelo_actualizado': False,
        'X_batch': [],
        'y_batch': [],
        'onnx_bytes': None,
        'jugadas_texto_data': ""  # Almacena el texto interno
    }
    
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = defaults[key]

# ========== INICIALIZACIÓN DE SESIÓN ==========
inicializar_session_state()

# Validación temprana del modelo
if not st.session_state.fase_inicial and st.session_state.modelo is None:
    st.error("❌ Modelo no entrenado: primero carga las 104 jugadas y entrena.")
    st.stop()

# ========== INTERFAZ PRINCIPAL ==========
st.title("🎯 Ruleta Americana: Sistema de Predicción Avanzado")

# Callback para sincronizar el widget con los datos internos
def update_jugadas_texto():
    st.session_state.jugadas_texto_data = st.session_state.jugadas_texto_widget

# ✅ 1. Opción para ingresar jugadas iniciales
with st.expander("📥 Ingresar primeras 104 jugadas manualmente"):
    # Widget con clave dedicada y callback
    jugadas_texto = st.text_area(
        "Pega aquí tus jugadas separadas por coma o espacio (mínimo 104 números):",
        placeholder="Ej: 17, 4, 26, 8, 32, 0, 36, ...",
        value=st.session_state.jugadas_texto_data,
        key='jugadas_texto_widget',
        on_change=update_jugadas_texto
    )
    
    if st.button("Cargar jugadas manuales"):
        try:
            # Usar el valor interno (jugadas_texto_data) en lugar del widget
            jugadas_validas = cargar_jugadas_desde_texto(st.session_state.jugadas_texto_data)
            
            if len(jugadas_validas) >= 104:
                # Agregar jugadas manteniendo límite de 500
                for jugada in jugadas_validas:
                    agregar_jugada_manual(jugada)
                
                # Actualizar estadísticas
                st.session_state.estadisticas_usuario = calcular_estadisticas(st.session_state.jugadas_usuario)
                
                st.success(f"✅ {len(jugadas_validas)} jugadas cargadas")
            else:
                st.warning(f"Debes ingresar al menos 104 jugadas válidas. Encontradas: {len(jugadas_validas)}")
        except Exception as e:
            st.error(f"Error al procesar jugadas: {e}")

# ---------- Flujo principal ----------
# Carga inicial de datos históricos
if st.session_state.fase_inicial:
    # Datos históricos ampliados (926 jugadas)
    historico = [
        # 15/07/2025 (129 jugadas)
        '26', '9', '4', '11', '24', '0', '20', '22', '13', '00', '30', '5', '21', '27', '7', '19', '12', '26', '9', '18', '29', '0', '8', '3', '1', '33', '28', '19', '30', '16', '14', '32', '4', '8', '1', '1', '17', '21', '5', '26', '23', '24', '30', '1', '31', '17', '22', '14', '15', '10', '00', '36', '36', '24', '25', '24', '10', '27', '9', '12', '29', '3', '29', '11', '32', '29', '6', '8', '21', '5', '22', '2', '8', '28', '00', '9', '30', '25', '00', '12', '36', '7', '35', '29', '24', '26', '14', '16', '0', '19', '22', '2', '22', '31', '22', '1', '10', '27', '27', '17', '34', '35', '19', '12', '3', '7', '7', '18', '20', '00', '29', '31', '5', '16', '29', '19', '0', '16', '17', '17', '13', '13', '33', '20', '16', '15', '21', '25', '8',
        
        # 14/07/2025 (204 jugadas)
        '25', '28', '33', '8', '27', '13', '0', '26', '12', '18', '23', '27', '20', '29', '17', '23', '21', '27', '32', '25', '24', '19', '35', '33', '12', '15', '19', '25', '11', '15', '7', '24', '31', '31', '36', '29', '1', '36', '28', '6', '31', '20', '8', '0', '26', '4', '18', '13', '16', '16', '13', '16', '23', '2', '21', '31', '12', '5', '21', '0', '35', '2', '24', '29', '21', '18', '28', '6', '18', '33', '35', '4', '35', '20', '26', '19', '27', '8', '15', '36', '8', '31', '13', '22', '33', '21', '28', '35', '21', '7', '14', '1', '24', '18', '25', '12', '36', '28', '0', '18', '0', '00', '4', '9', '6', '19', '4', '32', '28', '15', '8', '28', '9', '27', '34', '20', '12', '14', '14', '4', '36', '28', '25', '31', '28', '14', '8', '16', '13', '14', '24', '7', '00', '26', '24', '1', '25', '6', '10', '20', '2', '24', '17', '11', '33', '20', '14', '2', '26', '17', '2', '11', '30', '25', '32', '4', '18', '3', '14', '14', '15', '4', '31', '31', '28', '22', '36', '8', '19', '00', '32', '12', '3', '11', '33', '26', '14', '29', '25', '24', '31', '0', '36', '10', '27', '31', '30', '15', '29', '10', '8', '5', '1', '11', '34', '35', '0', '27', '14', '17', '3', '00', '36',
        
        # 17/07/2025 (104 jugadas)
        '23', '36', '10', '24', '2', '35', '8', '18', '17', '34', '33', '5', '21', '8', '00', '5', '25', '25', '18', '25', '33', '9', '28', '25', '7', '32', '28', '10', '1', '5', '26', '29', '34', '8', '25', '3', '1', '27', '6', '15', '18', '20', '35', '24', '35', '4', '7', '32', '0', '16', '00', '33', '26', '1', '2', '2', '15', '21', '19', '25', '11', '28', '21', '11', '1', '33', '20', '2', '32', '16', '24', '10', '28', '18', '23', '5', '23', '16', '28', '7', '13', '18', '34', '4', '19', '7', '22', '30', '21', '32', '32', '16', '25', '15', '31', '21', '10', '26', '1', '25', '24', '6', '00', '13',
        
        # 22/07/2025 (105 jugadas)
        '23', '00', '8', '2', '27', '24', '34', '5', '20', '18', '8', '10', '6', '2', '0', '25', '22', '20', '23', '6', '22', '23', '8', '2', '21', '25', '10', '2', '17', '18', '4', '00', '16', '0', '32', '5', '22', '28', '3', '24', '36', '3', '34', '5', '31', '30', '31', '6', '5', '36', '0', '29', '19', '24', '28', '24', '20', '3', '35', '33', '32', '31', '17', '17', '4', '5', '14', '35', '13', '12', '2', '24', '25', '6', '16', '24', '36', '18', '15', '12', '00', '00', '23', '30', '11', '7', '11', '00', '27', '32', '30', '4', '2', '19', '24', '7', '24', '18', '27', '31', '28', '32', '8', '33', '00',
        
        # 26/07/2025 (156 jugadas)
        '24', '19', '23', '14', '24', '31', '22', '14', '4', '6', '11', '35', '2', '12', '17', '33', '29', '3', '24', '8', '13', '31', '22', '2', '21', '13', '9', '00', '10', '7', '34', '1', '23', '5', '16', '5', '24', '20', '33', '34', '5', '25', '26', '26', '29', '26', '26', '13', '23', '27', '5', '14', '5', '17', '2', '17', '8', '24', '10', '0', '13', '29', '19', '9', '29', '20', '36', '33', '30', '2', '8', '4', '22', '6', '35', '7', '00', '26', '26', '17', '23', '28', '33', '32', '5', '21', '19', '35', '27', '1', '11', '0', '0', '13', '12', '13', '30', '23', '7', '34', '20', '16', '3', '33', '14', '00', '14', '29', '1', '20', '6', '28', '36', '8', '28', '35', '6', '18', '27', '21', '4', '3', '14', '3', '36', '26', '9', '1', '29', '35', '26', '27', '30', '27', '33', '31', '11', '27', '19', '00', '2', '5', '00', '32', '3', '28', '12', '33', '12', '33', '26', '34', '1', '20', '6', '19',
        
        # 27/07/2025 (228 jugadas)
        '15', '7', '4', '21', '9', '7', '0', '5', '34', '10', '19', '31', '9', '4', '22', '28', '14', '13', '36', '5', '3', '28', '8', '22', '28', '0', '26', '35', '21', '5', '32', '21', '26', '8', '27', '3', '31', '16', '36', '36', '14', '23', '28', '14', '19', '13', '20', '33', '2', '26', '9', '31', '25', '5', '22', '8', '15', '3', '25', '00', '2', '00', '12', '1', '12', '5', '3', '9', '1', '21', '17', '24', '29', '6', '0', '8', '16', '21', '19', '2', '21', '17', '13', '15', '29', '13', '34', '00', '23', '23', '16', '20', '14', '1', '8', '18', '2', '14', '14', '35', '8', '8', '20', '30', '29', '6', '22', '34', '5', '8', '19', '10', '19', '00', '1', '36', '33', '35', '16', '20', '14', '36', '1', '10', '17', '25', '12', '8', '33', '17', '00', '1', '30', '33', '7', '29', '00', '24', '19', '11', '14', '32', '14', '5', '11', '23', '0', '18', '29', '0', '31', '15', '10', '19', '7', '2', '30', '6', '35', '20', '6', '8', '32', '7', '23', '24', '26', '31', '9', '30', '25', '32', '5', '32', '4', '26', '27', '15', '12', '30', '20', '12', '26', '2', '3', '18', '16', '28', '16', '14', '23', '28', '4', '13', '10', '10', '3', '15', '32', '9', '24', '32', '33', '11', '10', '28', '4', '17', '31', '24', '7', '33', '10', '28', '7', '31', '29', '29', '36', '00', '36', '33', '1', '6', '20', '00', '26', '20'
    ]
    
    # Guardar datos históricos en estado de sesión
    st.session_state.base_historica = historico
    
    # Entrenar modelo con datos históricos
    if len(st.session_state.base_historica) >= 10 and st.session_state.modelo is None:
        with st.spinner("Entrenando modelo con datos históricos..."):
            model, pca, scaler, onnx_session, onnx_bytes = entrenar_modelo(st.session_state.base_historica)
            if model is not None:
                st.session_state.modelo = model
                st.session_state.pca = pca
                st.session_state.scaler = scaler
                st.session_state.onnx_session = onnx_session
                st.session_state.onnx_bytes = onnx_bytes
                st.session_state.fase_inicial = False
                st.success("✅ Modelo entrenado con 900+ jugadas históricas")
            else:
                st.error("Error al entrenar el modelo. Verifique los datos.")

# Interfaz de usuario
if st.session_state.fase_inicial:
    st.subheader("📌 Paso 1: Carga de datos iniciales")
    st.markdown("### 🎯 Carga 104 jugadas actuales para predicción en vivo")
    jugadas_actuales_input = st.text_area("Pega 104 jugadas actuales separadas por comas (ej: 17, 0, 23, 8, ...)")
    
    if st.button("📥 Cargar 104 jugadas actuales"):
        try:
            jugadas_validas = cargar_jugadas_desde_texto(jugadas_actuales_input)
            
            if not jugadas_validas:
                st.warning("No se detectaron jugadas válidas")
            else:
                # Agregar jugadas manteniendo límite de 500
                for jugada in jugadas_validas:
                    agregar_jugada_manual(jugada)
                
                # Inicializar estadísticas
                st.session_state.estadisticas_usuario = calcular_estadisticas(st.session_state.jugadas_usuario)
                
                # Entrenar modelo si hay suficientes jugadas
                if len(st.session_state.jugadas_usuario) >= 10:
                    with st.spinner("Entrenando modelo inicial..."):
                        model, pca, scaler, onnx_session, onnx_bytes = entrenar_modelo(st.session_state.jugadas_usuario)
                        if model is not None:
                            st.session_state.modelo = model
                            st.session_state.pca = pca
                            st.session_state.scaler = scaler
                            st.session_state.onnx_session = onnx_session
                            st.session_state.onnx_bytes = onnx_bytes
                            st.session_state.fase_inicial = False
                            st.success(f"✅ {len(jugadas_validas)} jugadas actuales cargadas. Modelo entrenado.")
                        else:
                            st.error("Error al entrenar el modelo. Verifique los datos.")
        except Exception as e:
            st.error(f"Error al cargar jugadas: {e}")

else:
    # Panel de control principal
    st.subheader("➕ Ingresar nueva jugada")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        nueva_jugada = st.selectbox("Selecciona el número que salió:", ruleta_americana, index=0)
    with col2:
        st.write("")
        st.write("")
        if st.button("Agregar jugada", key="add_btn", use_container_width=True):
            # Agregar manteniendo límite de 500
            agregar_jugada_manual(nueva_jugada)
            
            # Actualizar estadísticas
            actualizar_estadisticas_usuario(nueva_jugada)
        
            st.success(f"✅ {nueva_jugada} agregado a la historia")
            
            # Actualizar el área de texto con la nueva jugada
            actualizar_texto_jugadas(nueva_jugada)
            
            # Actualizar modelo con nuevo dato
            if len(st.session_state.jugadas_usuario) >= 11:
                ventana = st.session_state.jugadas_usuario[-11:-1]
                X_new = extraer_caracteristicas(ventana)
                actualizar_modelo_parcial(X_new, nueva_jugada)
    with col3:
        st.write("")
        st.write("")
        if st.button("Predecir sin agregar", key="predict_btn", use_container_width=True):
            pass  # Solo para forzar actualización
    
    # Mostrar recomendaciones
    if st.session_state.modelo and len(st.session_state.jugadas_usuario) >= 10:
        with st.spinner("Calculando predicciones..."):
            calientes, estrategicos, patrones = predecir_numeros()
        
        if calientes and estrategicos:
            st.subheader("🎯 Predicciones TOP 18")
            st.write(" → " + "  ".join(calientes + estrategicos))
            
            # Mostrar métricas de desempeño
            with st.expander("📈 Métricas del Modelo", expanded=True):
                mostrar_metricas_texto()
            
            # Mostrar patrones detectados
            with st.expander("🔍 Patrones detectados", expanded=False):
                if patrones:
                    for tipo, *desc in patrones:
                        st.markdown(f"**{tipo}**: {', '.join(desc)}")
                else:
                    st.info("No se detectaron patrones significativos")
            
            # Mostrar categorías de predicciones
            st.markdown("#### 🔥 9 Números Calientes")
            st.write(" ".join(calientes))
            
            st.markdown("#### 🎯 9 Números Estratégicos")
            st.write(" ".join(estrategicos))
        else:
            st.warning("No se pudieron generar predicciones. Verifica que el modelo esté entrenado correctamente.")
    elif st.session_state.modelo is None:
        st.warning("El modelo no está entrenado. Carga al menos 104 jugadas y entrena el modelo.")
    elif len(st.session_state.jugadas_usuario) < 10:
        st.warning("Se necesitan al menos 10 jugadas para realizar predicciones.")

    # Estadísticas y visualización
    st.subheader("📊 Análisis de Datos")
    
    with st.expander("Historial de Predicciones", expanded=False):
        mostrar_historial_simple()
    
    with st.expander("Últimas 20 Jugadas", expanded=False):
        if st.session_state.jugadas_usuario:
            ultimas_20 = st.session_state.jugadas_usuario[-20:]
            st.write(pd.DataFrame(ultimas_20, columns=["Jugada"]))
        else:
            st.info("No hay jugadas registradas")
    
    # Sección para cargar más jugadas actuales
    with st.expander("💾 Cargar más jugadas actuales", expanded=False):
        st.info("Ingresa jugadas adicionales para mejorar la precisión del modelo")
        mas_jugadas_input = st.text_area("Pega jugadas adicionales separadas por comas...", key="mas_jugadas")
        if st.button("Cargar jugadas adicionales", key="btn_mas_jugadas"):
            try:
                jugadas_validas = cargar_jugadas_desde_texto(mas_jugadas_input)
                
                if jugadas_validas:
                    # Agregar manteniendo límite de 500
                    for jugada in jugadas_validas:
                        agregar_jugada_manual(jugada)
                    
                    # Actualizar estadísticas
                    st.session_state.estadisticas_usuario = calcular_estadisticas(st.session_state.jugadas_usuario)
                    
                    # Actualizar modelo con nuevos datos
                    if len(st.session_state.jugadas_usuario) >= 11:
                        for i in range(10, len(st.session_state.jugadas_usuario)):
                            ventana = st.session_state.jugadas_usuario[i-10:i]
                            X_new = extraer_caracteristicas(ventana)
                            actualizar_modelo_parcial(X_new, st.session_state.jugadas_usuario[i])
                    st.success(f"✅ {len(jugadas_validas)} jugadas adicionales cargadas")
                else:
                    st.warning("No se detectaron jugadas válidas")
            except Exception as e:
                st.error(f"Error al cargar jugadas: {e}")

# Reinicio del sistema
st.sidebar.subheader("Administración del Sistema")
if st.sidebar.button("🔁 Reiniciar Todo", key="reset_btn", use_container_width=True):
    keys = list(st.session_state.keys())
    for key in keys:
        if key not in ['fase_inicial']:
            del st.session_state[key]
    
    # Reiniciar estado inicial
    inicializar_session_state()
    st.session_state.fase_inicial = True
    st.session_state.estadisticas_usuario = inicializar_estadisticas()
    st.success("✅ Sistema reiniciado correctamente")

# Información adicional
st.sidebar.markdown("### 🔍 Acerca del Sistema")
st.sidebar.info("""
**Sistema de predicción avanzado** que combina:
- Machine Learning (SGDClassifier optimizado)
- Inferencia ultra-rápida con ONNX
- Reducción de dimensionalidad (PCA)
- Actualizaciones parciales (partial_fit)
- Análisis estadístico con Z-Scores
- Patrones de experiencia
- Diversificación de apuestas
""")

st.sidebar.markdown("### ⚙️ Estado Actual")
st.sidebar.metric("Jugadas registradas", len(st.session_state.jugadas_usuario))

if st.session_state.jugadas_usuario:
    ultima_jugada = st.session_state.jugadas_usuario[-1] if st.session_state.jugadas_usuario else "-"
    st.sidebar.metric("Última jugada", ultima_jugada)
    
    # Mostrar propiedades de la última jugada
    props = number_properties.get(ultima_jugada, {})
    st.sidebar.metric("Color", props.get('color', ''))
    st.sidebar.metric("Paridad", props.get('parity', ''))
    st.sidebar.metric("Alto/Bajo", props.get('high_low', ''))
    st.sidebar.metric("Tipo de modelo", "SGD + ONNX")
    
# Validación de datos para predicción
if len(st.session_state.jugadas_usuario) < 10:
    st.sidebar.warning("⚠️ Agrega al menos 10 jugadas para predecir")
elif st.session_state.modelo is None:
    st.sidebar.error("❌ Modelo no entrenado")

# Exportar datos
st.sidebar.download_button(
    label="📤 Exportar Datos",
    data=pd.DataFrame(st.session_state.jugadas_usuario, columns=["Jugada"]).to_csv().encode('utf-8'),
    file_name=f"ruleta_data_{time.strftime('%Y%m%d-%H%M%S')}.csv",
    mime='text/csv',
    use_container_width=True
)