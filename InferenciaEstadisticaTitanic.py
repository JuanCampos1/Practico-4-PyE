import scipy.stats as stats
import numpy as np

class InferenciaEstadisticaTitanic:
    def __init__(self, df):
        self.df = df

    def intervaloConfianzaEdad(self, confianza=0.95):
        media = np.mean(self.df['age'])
        desviacionEstandar = np.std(self.df['age'], ddof=1)
        n = len(self.df['age'])
        h = stats.t.ppf((1 + confianza) / 2, n - 1) * (desviacionEstandar / np.sqrt(n))
        return media - h, media + h

    def pruebaPromedioEdad(self, genero, valorHipotetico, confianza=0.95):
        edades = self.df[self.df['gender'] == genero]['age']
        tStat, pValue = stats.ttest_1samp(edades, valorHipotetico)
        return tStat, pValue / 2  # Usamos /2 para la prueba unilateral

    def diferenciaSupervivenciaGenero(self, confianza=0.99):
        hombres = self.df[self.df['gender'] == 'male']['survived']
        mujeres = self.df[self.df['gender'] == 'female']['survived']
        tStat, pValue = stats.ttest_ind(hombres, mujeres)
        return tStat, pValue

    def diferenciaSupervivenciaClase(self, confianza=0.99):
        clases = [self.df[self.df['p_class'] == i]['survived'] for i in range(1, 4)]
        fStat, pValue = stats.f_oneway(*clases)
        return fStat, pValue

    def pruebaPromedioEdadGenero(self, confianza=0.95):
        edadesHombres = self.df[self.df['gender'] == 'male']['age']
        edadesMujeres = self.df[self.df['gender'] == 'female']['age']
        tStat, pValue = stats.ttest_ind(edadesMujeres, edadesHombres)
        return tStat, pValue
