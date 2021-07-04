**Objetivo General:**
Obtener un modelo con BERT que ante un texto y una pregunta pueda predecir dónde está la respuesta. Como baseline se debe usar BERT pre-entrenado, luego aplicar alguna modificación al modelo/entrenamiento/preprocesado para mejorar este baseline.

# Datasets
Se utilizaron 6 datasets, 3 de conocimiento general (SQuAD, NewsQA y Natural Questions) y 3 de conocimiento específico (RACE, DuoRC y RelationExtraction)
*SQuAD:* desc - train - val
*NewsQA:* desc - train - val
*Natural Questions:* desc - train - val
*RACE:* desc - train - val
*DuoRC:* desc - train - val
*RelationExtraction:* desc - train - val

# Preprocesado
Los ejemplos de los datasets, tienen el siguiente formato:

Para el preprocesado, se parseó correctamente cada conjunto de contexto-pregunta-respuesta para luego encontrar las posiciones de los indices de los strings ya tokenizados en donde comienza la respuesta y en donde termina.

Se crearon dataloaders con los datasets mezclados para alimentar al modelo durante el entrenamiento, validación y prueba.

# Baseline
Como primera aproximación al problema, se utilizó a DistilBert-base-cased como modelo pre-entrenado con dos MLP en la salida, uno para la predicción del token de comienzo de respuesta y otro para el token de fin.

RESULTADOS
tiempo, F1, EM, gráficos

# Mejoras

# Resultados
