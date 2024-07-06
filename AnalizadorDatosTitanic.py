import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AnalizadorDatosTitanic:
    def __init__(self, rutaArchivo):
        self.df = pd.read_csv(rutaArchivo)

    def completarEdadFaltante(self):
        mediaEdadporGenero = self.df.groupby('gender')['age'].mean()
        self.df['age'] = self.df.apply(lambda row: mediaEdadporGenero[row['gender']] if pd.isnull(row['age']) else row['age'], axis=1)

    def calcularEstadísticasDescriptivas(self):
        mediaEdad = self.df['age'].mean()
        medianaEdad = self.df['age'].median()
        modaEdad = self.df['age'].mode()[0]
        rangoEdad = self.df['age'].max() - self.df['age'].min()
        varianzaEdad = self.df['age'].var()
        desviacionEdad = self.df['age'].std()

        return mediaEdad, medianaEdad, modaEdad, rangoEdad, varianzaEdad, desviacionEdad

    def calcularTasaSupervivencia(self):
        pasajerosTotales = len(self.df)
        sobrevivientes = self.df['survived'].sum()
        tasaSupervivencia = sobrevivientes / pasajerosTotales
        return tasaSupervivencia

    def graficarHistogramasEdad(self):
        bins = range(int(self.df['age'].min()), int(self.df['age'].max()) + 1)
        plt.figure(figsize=(15, 5))
        for i in range(1, 4):
            plt.subplot(1, 3, i)
            plt.hist(self.df[self.df['p_class'] == i]['age'], bins=bins, alpha=0.5, edgecolor='black', linewidth=1.2)
            plt.xlabel('Edad')
            plt.ylabel('Frecuencia')
            plt.title(f'Clase {i}')
        plt.tight_layout()
        plt.show()

    def graficarDiagramaCajasEdad(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='survived', y='age', data=self.df)
        plt.xlabel('Sobrevivientes')
        plt.ylabel('Edad')
        plt.title('Diagrama de Cajas de las Edades de los Supervivientes y No Supervivientes')
        plt.xticks([0, 1], ['No', 'Sí'])
        plt.show()