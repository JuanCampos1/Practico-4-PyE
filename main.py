from AnalizadorDatosTitanic import AnalizadorDatosTitanic
from InferenciaEstadisticaTitanic import InferenciaEstadisticaTitanic

# Crear instancia del analizador de datos y ejecutar análisis
analizador = AnalizadorDatosTitanic('titanik.csv')
analizador.completarEdadFaltante()
mediaEdad, medianaEdad, modaEdad, rangoEdad, varianzaEdad, desviacionEdad = analizador.calcularEstadísticasDescriptivas()
tasaSupervivencia = analizador.calcularTasaSupervivencia()
analizador.graficarHistogramasEdad()
analizador.graficarDiagramaCajasEdad()

# Mostrar resultados
print(f"Media de las edades: {mediaEdad}")
print(f"Mediana de las edades: {medianaEdad}")
print(f"Moda de las edades: {modaEdad}")
print(f"Rango de las edades: {rangoEdad}")
print(f"Varianza de las edades: {varianzaEdad}")
print(f"Desviación estándar de las edades: {desviacionEdad}")
print(f"Tasa de supervivencia general: {tasaSupervivencia}")

# Crear instancia para inferencia estadística
inferencia = InferenciaEstadisticaTitanic(analizador.df)

# Paso 2.1: Intervalo de confianza para la edad promedio
intervaloConfianza = inferencia.intervaloConfianzaEdad()
print(f"Intervalo de confianza para la edad promedio (95%): {intervaloConfianza}")

# Paso 2.2: Prueba de hipótesis para edad promedio de mujeres y hombres
tStatMujeres, pValueMujeres = inferencia.pruebaPromedioEdad('female', 56)
print(f"Prueba de hipótesis para edad promedio de mujeres > 56: t={tStatMujeres}, p={pValueMujeres}")

tStatHombres, pValueHombres = inferencia.pruebaPromedioEdad('male', 56)
print(f"Prueba de hipótesis para edad promedio de hombres > 56: t={tStatHombres}, p={pValueHombres}")

# Paso 2.3: Diferencia significativa en la tasa de supervivencia entre géneros y clases
tStatGenero, pValueGenero = inferencia.diferenciaSupervivenciaGenero()
print(f"Diferencia en supervivencia entre géneros: t={tStatGenero}, p={pValueGenero}")

fStatClase, pValueClase = inferencia.diferenciaSupervivenciaClase()
print(f"Diferencia en supervivencia entre clases: F={fStatClase}, p={pValueClase}")

# Paso 2.4: Prueba de hipótesis para diferencias en edades entre géneros
tStatEdad, pValueEdad = inferencia.pruebaPromedioEdadGenero()
print(f"Diferencia en edades entre géneros: t={tStatEdad}, p={pValueEdad}")