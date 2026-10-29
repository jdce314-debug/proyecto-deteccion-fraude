
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min as min_value,
    max as max_value,
    sum as sum_value,
    when,
    round as spark_round
)

# ==================================================
# 1. CONFIGURAR SPARK
# ==================================================

spark = (
    SparkSession.builder
    .appName("DeteccionFraudeTarjetas")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n========================================")
print(" PROYECTO: DETECCION DE FRAUDE CON SPARK")
print("========================================\n")

# ==================================================
# 2. LOCALIZAR Y CARGAR EL DATASET
# ==================================================

carpeta = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(carpeta, "creditcard.csv")
salida = os.path.join(carpeta, "resultados_spark")

os.makedirs(salida, exist_ok=True)

try:
    if not os.path.exists(archivo):
        raise FileNotFoundError(
            "No se encontro creditcard.csv. "
            "Coloca el archivo en la misma carpeta "
            "que este programa."
        )

    print("PASO 1: Cargando el conjunto de datos...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(archivo)
    )

    print("Datos cargados correctamente.")
    print("Numero de filas:", df.count())
    print("Numero de columnas:", len(df.columns))

    print("\nColumnas del conjunto de datos:")
    df.printSchema()

    print("\nPrimeras cinco transacciones:")
    df.show(5, truncate=False)

    # ==================================================
    # 3. LIMPIEZA DE DATOS
    # ==================================================

    print("\nPASO 2: Revisando la calidad de los datos...")

    print("\nValores nulos por columna:")
    df.select([
        sum_value(
            when(col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df.columns
    ]).show(truncate=False)

    filas_originales = df.count()
    duplicados = filas_originales - df.dropDuplicates().count()

    print("Filas duplicadas encontradas:", duplicados)

    df_limpio = df.dropDuplicates()

    columnas_numericas = [
        c for c in df_limpio.columns if c != "Class"
    ]

    for c in columnas_numericas:
        df_limpio = df_limpio.withColumn(
            c, col(c).cast("double")
        )

    df_limpio = df_limpio.withColumn(
        "Class", col("Class").cast("integer")
    )

    df_limpio = df_limpio.dropna(
        subset=df_limpio.columns
    )

    df_limpio = df_limpio.filter(
        col("Class").isin(0, 1)
    )

    print("Filas despues de la limpieza:", df_limpio.count())

    print("\nDistribucion de la variable Class:")
    df_limpio.groupBy("Class").count().show()

    # Guardar datos limpios
    df_limpio.write.mode("overwrite").parquet(
        os.path.join(salida, "datos_limpios")
    )

    # ==================================================
    # 4. ANALISIS EXPLORATORIO DE DATOS (EDA)
    # ==================================================

    print("\nPASO 3: Analisis exploratorio de datos (EDA)")

    print("\nResumen estadistico de los datos:")
    df_limpio.describe(
        ["Time", "Amount"]
    ).show()

    print("\nCantidad de transacciones por clase:")

    resumen = (
        df_limpio.groupBy("Class")
        .agg(
            count("*").alias("Cantidad"),
            spark_round(avg("Amount"), 2).alias(
                "Promedio_Amount"
            ),
            spark_round(min_value("Amount"), 2).alias(
                "Monto_Minimo"
            ),
            spark_round(max_value("Amount"), 2).alias(
                "Monto_Maximo"
            )
        )
        .orderBy("Class")
    )

    total = df_limpio.count()

    resumen = resumen.withColumn(
        "Porcentaje",
        spark_round(col("Cantidad") * 100 / total, 4)
    )

    resumen.show(truncate=False)

    resumen.write.mode("overwrite").option(
        "header", True
    ).csv(os.path.join(salida, "resumen_por_clase"))

    # ==================================================
    # 5. CLASIFICAR LOS MONTOS DE LAS TRANSACCIONES
    # ==================================================

    print("\nPASO 4: Agrupando transacciones por monto...")

    montos = df_limpio.withColumn(
        "Rango_monto",
        when(col("Amount") == 0, "Monto cero")
        .when(col("Amount") < 50, "Menos de 50")
        .when(col("Amount") < 200, "De 50 a menos de 200")
        .otherwise("200 o mas")
    )

    rangos = (
        montos.groupBy("Rango_monto", "Class")
        .count()
        .orderBy("Rango_monto", "Class")
    )

    rangos.show(truncate=False)

    rangos.write.mode("overwrite").option(
        "header", True
    ).csv(os.path.join(salida, "rangos_monto"))

    # ==================================================
    # 6. GUARDAR UNA MUESTRA DE RESULTADOS
    # ==================================================

    print("\nPASO 5: Guardando una muestra de los resultados...")

    df_limpio.limit(1000).coalesce(1).write.mode(
        "overwrite"
    ).option("header", True).csv(
        os.path.join(salida, "muestra_datos_limpios")
    )

    # ==================================================
    # 7. VISUALIZAR RESULTADOS
    # ==================================================

    print("\nPASO 6: Generando graficas...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clases = (
            df_limpio.groupBy("Class")
            .count()
            .orderBy("Class")
            .collect()
        )

        nombres = [
            "Transacciones legitimas" if r["Class"] == 0
            else "Fraudes"
            for r in clases
        ]

        cantidades = [r["count"] for r in clases]

        plt.figure(figsize=(8, 5))
        plt.bar(nombres, cantidades)
        plt.title("Transacciones legitimas y fraudulentas")
        plt.xlabel("Tipo de transaccion")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig(
            os.path.join(salida, "transacciones_por_clase.png")
        )
        plt.close()

        promedios = (
            df_limpio.groupBy("Class")
            .agg(avg("Amount").alias("Promedio"))
            .orderBy("Class")
            .collect()
        )

        nombres_monto = [
            "Legitimas" if r["Class"] == 0 else "Fraudes"
            for r in promedios
        ]

        valores_monto = [r["Promedio"] for r in promedios]

        plt.figure(figsize=(8, 5))
        plt.bar(nombres_monto, valores_monto)
        plt.title("Monto promedio por tipo de transaccion")
        plt.xlabel("Tipo de transaccion")
        plt.ylabel("Monto promedio")
        plt.tight_layout()
        plt.savefig(
            os.path.join(salida, "monto_promedio.png")
        )
        plt.close()

        print("Graficas guardadas correctamente.")

    except Exception as error_grafica:
        print("No fue posible generar las graficas:", error_grafica)

    # ==================================================
    # 8. RESUMEN FINAL
    # ==================================================

    print("\n========================================")
    print(" PROCESAMIENTO FINALIZADO")
    print("========================================")
    print("Filas originales:", filas_originales)
    print("Duplicados eliminados:", duplicados)
    print("Filas finales:", df_limpio.count())
    print("Resultados guardados en:", salida)
    print("\nRevisa la carpeta resultados_spark.")
    print("========================================\n")

except Exception as error:
    print("\nERROR DURANTE EL PROCESAMIENTO:")
    print(error)
    sys.exit(1)

finally:
    spark.stop()
