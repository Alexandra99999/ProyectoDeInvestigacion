import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configuración para ignorar warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.random.seed(42)
plt.style.use('ggplot')
sns.set_palette('viridis')

# Crear directorio para guardar resultados si no existe
output_dir = 'resultados_analisis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =============================================================================
# Procesamiento de los datos del Excel
# =============================================================================
data = {
    'Codigo': [
        71231040, 71231041, 71231074, 71231045, 71231051, 71231071, 71231036,
        71231028, 71231004, 71231033, 71221106, 71231002, 71231008, 71231038,
        71231024, 71231053, 71231070, 71231019, 71231007, 71231008,

        80000001, 80000002, 80000003, 80000004, 80000005,
        80000006, 80000007, 80000008, 80000009, 80000010,
        80000011, 80000012, 80000013, 80000014, 80000015
    ],
    'Estudiante': [
        'Calambas Epe Andrés Felipe',
        'Campo Chuga Milton Andres',
        'Campo Palta Rosana',
        'Giraldo Muñoz Juan Diego',
        'Grueso Gómez Ronald Fernando',
        'Imbachi Granda Aleks Steban',
        'Manquillo Lara Juan Sebastián',
        'Martinez Ordoñez Karen Tatiana',
        'Martinez Ortiz Juan Camilo',
        'Navia Gomez Freddy Andres',
        'Paz Jimenez Juan Esteban',
        'Quinayas Jimenez Ady',
        'Restrepo Muñoz Yeison Daniel',
        'Rosero Caicedo Jonier Mauricio',
        'Sanchez Villaquiran Brayan José',
        'Tocoche Quintero Jhon Alexander',
        'Villamarin Flor Eymer Daniel',
        'Zambrano Bolaños Alexis',
        'Muñoz Carvajal Sebastian',
        'Mogollón Alegría Yilmar',

        'Lopez Perez Mauricio',
        'Garcia Ruiz Juan Jose',
        'Torres Mina Oscar',
        'Ramos Castillo Pedro',
        'Vargas Peña Jennifer',
        'Moreno Silva Patricia',
        'Ortega Rojas Olga',
        'Castro Medina Julian',
        'Suarez Pardo Carlos',
        'Herrera Quintero Jose',

        'Martinez Jose Dario',
        'Vidal Carla',
        'Perez Juan',
        'Rebolledo Martha',
        'Lopez Angela'
    ],
    'Nota_Corte1': [
        4.336, 3.2, 4.856, 4.688000000000001, 4.336, 3.2, 4.688000000000001,
        4.856, 4.856, 4.856, 0, 4.336, 3.2, 3.2, 4.688000000000001, 4.336,
        4.688000000000001, 4.336, 4.688000000000001, 3.2,

        2.1, 1.8, 2.5, 2.0, 1.6,
        2.3, 1.9, 2.7, 2.4, 1.5,
        3.0, 3.1, 3.2, 2.9, 3.3
    ],
    'Nota_Corte2': [
        4.340000000000001, 3.62, 4.300000000000001, 4.88, 4.340000000000001, 3.62,
        4.960000000000001, 4.26, 4.42, 4.34, 0, 4.340000000000001,
        3.8600000000000003, 3.8600000000000003, 4.68, 4.340000000000001,
        4.640000000000001, 4.340000000000001, 4.960000000000001, 4.44,

        2.0, 1.7, 2.3, 2.1, 1.8,
        2.2, 1.6, 2.5, 2.2, 1.4,
        3.2, 3.0, 3.3, 3.1, 3.4
    ],
    'Nota_Corte3': [
        3.6880000000000006, 4.808, 4.76, 4.76, 3.6880000000000006, 4.76, 4.952,
        4.76, 4.76, 4.76, 0, 3.6880000000000006, 3.6880000000000006,
        3.6880000000000006, 4.76, 3.6880000000000006, 4.76, 3.6880000000000006, 5, 4.76,

        2.2, 1.9, 2.4, 2.0, 1.7,
        2.1, 1.8, 2.6, 2.3, 1.6,
        3.1, 3.2, 3.4, 3.0, 3.5
    ],
    'Materia': [
    'ENFOQUE EMPRESARIAL',
    'TEORIA GENERAL DE SISTEMAS',
    'INGENIERIA DEL SOFTWARE II',
    'CALIDAD DEL SOFTWARE',
    'GESTION DE PROYECTOS INFORMÁTICOS',
    'FUNDAMENTOS DE LOS SISTEMAS DE INFORMACIÓN',
    'ALGORITMOS',
    'PROGRAMACIÓN ORIENTADA A OBJETOS I',
    'PROGRAMACIÓN ORIENTADA A OBJETOS II',
    'ESTRUCTURAS DE DATOS',
    'BASES DE DATOS',
    'INGENIERÍA DEL SOFTWARE I',
    'ADMINISTRACIÓN DE BASES DE DATOS',
    'ALGORITMOS COMPUTACIONALES',
    'APLICACIONES WEB',

    # Repetición para cubrir todos los estudiantes
    'ENFOQUE EMPRESARIAL',
    'TEORIA GENERAL DE SISTEMAS',
    'INGENIERIA DEL SOFTWARE II',
    'CALIDAD DEL SOFTWARE',
    'GESTION DE PROYECTOS INFORMÁTICOS',

    'FUNDAMENTOS DE LOS SISTEMAS DE INFORMACIÓN',
    'ALGORITMOS',
    'PROGRAMACIÓN ORIENTADA A OBJETOS I',
    'PROGRAMACIÓN ORIENTADA A OBJETOS II',
    'ESTRUCTURAS DE DATOS',
    'BASES DE DATOS',
    'INGENIERÍA DEL SOFTWARE I',
    'ADMINISTRACIÓN DE BASES DE DATOS',
    'ALGORITMOS COMPUTACIONALES',
    'APLICACIONES WEB',

    'ENFOQUE EMPRESARIAL',
    'TEORIA GENERAL DE SISTEMAS',
    'INGENIERIA DEL SOFTWARE II',
    'CALIDAD DEL SOFTWARE',
    'GESTION DE PROYECTOS INFORMÁTICOS'
]
}

df = pd.DataFrame(data)

# Calcular nota final
df['Nota_Final'] = 0.3*df['Nota_Corte1'] + 0.3*df['Nota_Corte2'] + 0.4*df['Nota_Corte3']

# =============================================================================
# Generación de datos sintéticos
# =============================================================================
def generate_synthetic_data(original_df, n_synthetic=2000):
    synthetic_data = []
    cortes = ['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3']
    
    stats = {}
    for corte in cortes:
        std_val = max(original_df[corte].std(), 0.5)
        stats[corte] = {
            'mean': original_df[corte].mean(),
            'std': std_val,
            'min': original_df[corte].min(),
            'max': original_df[corte].max()
        }
    
    for _ in range(n_synthetic):
        notas = {}
        for corte in cortes:
            nota = np.random.normal(stats[corte]['mean'], stats[corte]['std'] * 1.5)
            nota = np.clip(nota, 0, 5)
            
            rand_val = np.random.rand()
            if rand_val < 0.15:
                nota = np.clip(nota - np.random.uniform(1.0, 2.5), 0, 5)
            elif rand_val < 0.35:
                nota = np.clip(nota + np.random.uniform(0.5, 1.5), 0, 5)
                
            notas[corte] = nota
        
        nota_final = 0.3*notas['Nota_Corte1'] + 0.3*notas['Nota_Corte2'] + 0.4*notas['Nota_Corte3']
        
        synthetic_data.append({
            'Nota_Corte1': notas['Nota_Corte1'],
            'Nota_Corte2': notas['Nota_Corte2'],
            'Nota_Corte3': notas['Nota_Corte3'],
            'Nota_Final': nota_final,
            'Tipo': 'Sintético'
        })
    
    return pd.DataFrame(synthetic_data)

synthetic_df = generate_synthetic_data(df, n_synthetic=2000)
df['Tipo'] = 'Original'
combined_df = pd.concat([df, synthetic_df], ignore_index=True)

# Guardar datos combinados en CSV
#combined_df.to_csv(f'{output_dir}/datos_combinados.csv', index=False)

# =============================================================================
# Cálculo de probabilidades
# =============================================================================
corte_stats = {}
for corte in ['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3']:
    std_val = combined_df[corte].std()
    if std_val < 0.1:
        std_val = 0.5
    corte_stats[corte] = {'mean': combined_df[corte].mean(), 'std': std_val}

nota_final_std = max(combined_df['Nota_Final'].std(), 0.5)

def calculate_probability(nota, std):
    safe_std = max(std, 0.5)
    return 1 - norm.cdf(3.0, loc=nota, scale=safe_std)

prob_cols = []
for corte in ['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3', 'Nota_Final']:
    col_name = f'Prob_{corte}'
    prob_cols.append(col_name)
    
    if corte == 'Nota_Final':
        std = nota_final_std
    else:
        std = corte_stats[corte]['std']
    
    df[col_name] = df[corte].apply(lambda x: calculate_probability(x, std))

df['Objetivo_Cumplido'] = df['Prob_Nota_Final'] >= 0.5

# Guardar resultados en CSV
df.to_csv(f'{output_dir}/resultados_analisis.csv', index=False)

# =============================================================================
# Visualizaciones completas
# =============================================================================

# 1. Distribución de notas
plt.figure(figsize=(12, 8))
sns.boxplot(data=combined_df[['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3', 'Nota_Final']],
            palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Distribución de Notas por Corte y Final', fontsize=16)
plt.ylabel('Nota', fontsize=14)
plt.axhline(y=3.0, color='r', linestyle='--', alpha=0.7)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_distribucion_notas.png', dpi=300)
#plt.show()

# 2. Probabilidad de aprobación por corte
plt.figure(figsize=(12, 8))
prob_data = df[prob_cols].melt(var_name='Corte', value_name='Probabilidad')
sns.boxplot(x='Corte', y='Probabilidad', data=prob_data)
plt.title('Probabilidad de Aprobación por Corte', fontsize=16)
plt.xticks(ticks=[0, 1, 2, 3], 
           labels=['Corte 1', 'Corte 2', 'Corte 3', 'Final'], 
           fontsize=12)
plt.ylabel('Probabilidad', fontsize=14)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_probabilidad_por_corte.png', dpi=300)
#plt.show()

# 3. Relación entre notas y probabilidad
plt.figure(figsize=(14, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, corte in enumerate(['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3', 'Nota_Final']):
    sns.scatterplot(x=corte, y=f'Prob_{corte}', data=df, s=100, alpha=0.8, 
                    color=colors[i], label=corte.replace('_', ' '))
    
    filtered = df.dropna(subset=[corte, f'Prob_{corte}'])
    if len(filtered) > 3:
        x_vals = filtered[corte].values
        y_vals = filtered[f'Prob_{corte}'].values
        x_range = np.ptp(x_vals)
        if x_range < 0.1:
            coef = np.polyfit(x_vals, y_vals, 1)
            poly1d_fn = np.poly1d(coef)
            x_line = np.linspace(min(x_vals), max(x_vals), 100)
            plt.plot(x_line, poly1d_fn(x_line), '--', lw=3, color=colors[i])
        else:
            try:
                lowess_smoothed = lowess(y_vals, x_vals, frac=0.5, it=2)
                plt.plot(lowess_smoothed[:,0], lowess_smoothed[:,1], '--', lw=3, color=colors[i])
            except:
                coef = np.polyfit(x_vals, y_vals, 1)
                poly1d_fn = np.poly1d(coef)
                x_line = np.linspace(min(x_vals), max(x_vals), 100)
                plt.plot(x_line, poly1d_fn(x_line), '--', lw=3, color=colors[i])
    
plt.title('Relación entre Notas y Probabilidad de Aprobar', fontsize=16)
plt.xlabel('Nota', fontsize=14)
plt.ylabel('Probabilidad de Aprobación', fontsize=14)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=3.0, color='r', linestyle='--', alpha=0.7)
plt.legend(title='Corte', fontsize=12, title_fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(f'{output_dir}/3_relacion_notas_probabilidad.png', dpi=300)
#plt.show()

# 4. Distribución normal por corte
plt.figure(figsize=(12, 8))
cortes = ['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, corte in enumerate(cortes):
    mean = corte_stats[corte]['mean']
    std = corte_stats[corte]['std']
    x = np.linspace(0, 5, 500)
    y = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, y, label=f'{corte.replace("_", " ")} (μ={mean:.2f}, σ={std:.2f})', 
             color=colors[i], lw=3)
    plt.fill_between(x, y, where=(x>=3), color=colors[i], alpha=0.2)

plt.title('Modelo de Distribución Normal por Corte', fontsize=16)
plt.xlabel('Nota', fontsize=14)
plt.ylabel('Densidad de Probabilidad', fontsize=14)
plt.axvline(x=3.0, color='r', linestyle='--', label='Umbral de aprobación')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/4_distribucion_normal_cortes.png', dpi=300)
#plt.show()

# 5. Probabilidades de estudiantes destacados
plt.figure(figsize=(14, 10))
# Seleccionar estudiantes con notas extremas
high_risk = df[df['Prob_Nota_Final'] < 0.5]
high_perf = df[df['Prob_Nota_Final'] > 0.95]
sample_df = pd.concat([high_risk, high_perf]).sample(5, random_state=42)

sample_data = sample_df[['Estudiante'] + prob_cols].melt(
    id_vars='Estudiante', 
    var_name='Corte', 
    value_name='Probabilidad'
)

# Acortar nombres de estudiantes
sample_data['Estudiante'] = sample_data['Estudiante'].apply(
    lambda x: x[:15] + '...' if len(x) > 15 else x
)

sns.barplot(x='Probabilidad', y='Estudiante', hue='Corte', data=sample_data, 
            palette='viridis', hue_order=prob_cols)
plt.title('Probabilidad de Aprobar por Estudiante (Muestra Representativa)', fontsize=16)
plt.xlabel('Probabilidad', fontsize=14)
plt.ylabel('Estudiante', fontsize=14)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
plt.legend(title='Corte', fontsize=12, title_fontsize=12, 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
#plt.savefig(f'{output_dir}/5_probabilidad_estudiantes.png', dpi=300)
#plt.show()

# 6. Matriz de correlación entre cortes
plt.figure(figsize=(12, 10))
corr_matrix = combined_df[['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3', 'Nota_Final']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True,
            annot_kws={"size": 14}, linewidths=0.5)
plt.title('Correlación entre Cortes y Nota Final', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/6_correlacion_cortes.png', dpi=300)
#plt.show()

# 7. Evolución de notas para estudiantes con riesgo
plt.figure(figsize=(12, 8))
# Identificar estudiantes con probabilidad final < 0.5
risk_df = df[df['Prob_Nota_Final'] < 0.5].copy()

if not risk_df.empty:
    # Seleccionar hasta 5 estudiantes para mostrar
    risk_sample = risk_df.head(5)
    
    for _, row in risk_sample.iterrows():
        cortes = [row['Nota_Corte1'], row['Nota_Corte2'], row['Nota_Corte3'], row['Nota_Final']]
        plt.plot(['Corte 1', 'Corte 2', 'Corte 3', 'Final'], cortes, 
                 marker='o', markersize=10, lw=2.5, 
                 label=f"{row['Estudiante'][:15]}... (Final: {row['Nota_Final']:.2f})")
    
    plt.axhline(y=3.0, color='r', linestyle='--', alpha=0.7)
    plt.title('Evolución de Notas (Estudiantes con Riesgo de Reprobación)', fontsize=16)
    plt.ylabel('Nota', fontsize=14)
    plt.ylim(0, 5.5)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
else:
    plt.text(0.5, 0.5, '¡Todos los estudiantes tienen\nprobabilidad de aprobación >50%!', 
             ha='center', va='center', fontsize=16)
    plt.title('No hay estudiantes con riesgo de reprobar', fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/7_evolucion_notas_riesgo.png', dpi=300)
#plt.show()

# 8. Distribución de probabilidades finales
plt.figure(figsize=(12, 8))
sns.histplot(df['Prob_Nota_Final'], kde=True, bins=20, color='skyblue', alpha=0.7)
plt.axvline(x=0.5, color='r', linestyle='--', lw=2, label='Umbral de aprobación')
plt.title('Distribución de Probabilidades de Aprobación Final', fontsize=16)
plt.xlabel('Probabilidad de Aprobar', fontsize=14)
plt.ylabel('Número de Estudiantes', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/8_distribucion_probabilidades.png', dpi=300)
#plt.show()

# =============================================================================
# Resultados finales
# =============================================================================
result_cols = ['Codigo', 'Estudiante', 'Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3', 'Nota_Final',
               'Prob_Nota_Corte1', 'Prob_Nota_Corte2', 'Prob_Nota_Corte3', 'Prob_Nota_Final',
               'Objetivo_Cumplido']

# Formatear probabilidades como porcentaje
for col in prob_cols:
    df[col] = df[col].apply(lambda x: f"{x:.2%}")

print("\n" + "="*90)
print("Resultados de Predicción para Estudiantes")
print("="*90)
print(df[result_cols].to_string(index=False))

# Resumen estadístico
print("\nResumen Estadístico:")
print(f"- Estudiantes analizados: {len(df)}")
print(f"- Porcentaje con objetivo cumplido: {df['Objetivo_Cumplido'].mean():.2%}")
print(f"- Promedio nota final: {df['Nota_Final'].mean():.2f}")
print(f"- Desviación estándar nota final: {nota_final_std:.2f}")
print(f"- Estudiantes en riesgo: {len(risk_df)}")

print(f"\nResultados guardados en el directorio: '{output_dir}'")
#print(f"- Datos completos: {output_dir}/datos_combinados.csv")
print(f"- Resultados análisis: {output_dir}/resultados_analisis.csv")
print(f"- Gráficas: {output_dir}/*.png")

# =============================================================================
# Intervención por corte y nota final (ALTO + MEDIO)
# =============================================================================

print("\n" + "="*70)
print("ESTUDIANTES EN INTERVENCIÓN (POR CORTE, NOTA FINAL Y MATERIA)")
print("="*70)

cortes = ['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3', 'Nota_Final']

def prob_to_float(col):
    return df[col].str.replace('%','').astype(float)/100

for corte in cortes:
    df[f'{corte}_num_prob'] = prob_to_float(f'Prob_{corte}')

def clasificar_riesgo(p):
    if p < 0.3:
        return 'ALTO'
    elif p < 0.5:
        return 'MEDIO'
    else:
        return 'BAJO'

for _, row in df.iterrows():
    
    riesgos_altos = []
    riesgos_medios = []
    
    for c in cortes:
        p = row[f'{c}_num_prob']
        nivel = clasificar_riesgo(p)
        
        if nivel == 'ALTO':
            riesgos_altos.append(c.replace('_', ' '))
        elif nivel == 'MEDIO':
            riesgos_medios.append(c.replace('_', ' '))
    
    if riesgos_altos or riesgos_medios:
        
        if len(riesgos_altos) >= 3:
            tipo = "CRÍTICO"
        elif len(riesgos_altos) >= 1:
            tipo = "ALTO"
        elif len(riesgos_medios) >= 2:
            tipo = "MEDIO"
        else:
            tipo = "BAJO-MEDIO"
        
        print(f"\nEstudiante: {row['Estudiante']}")
        print(f"Materia: {row['Materia']}")
        print(f"Nota Final: {row['Nota_Final']:.2f}")
        print(f"Tipo de Riesgo: {tipo}")
        
        if riesgos_altos:
            print(f"Riesgo ALTO en: {', '.join(riesgos_altos)}")
        
        if riesgos_medios:
            print(f"Riesgo MEDIO en: {', '.join(riesgos_medios)}")
        
        print("-"*50)

# Evaluar riesgos
df['Total_Riesgos_Altos'] = 0
df['Total_Riesgos_Medios'] = 0

# Mostrar resultados
for _, row in df.iterrows():
    
    riesgos_altos = []
    riesgos_medios = []
    
    for c in cortes:
        p = row[f'{c}_num_prob']
        nivel = clasificar_riesgo(p)
        
        if nivel == 'ALTO':
            riesgos_altos.append(c.replace('_', ' '))
        elif nivel == 'MEDIO':
            riesgos_medios.append(c.replace('_', ' '))
    
    # Mostrar solo si tiene algún riesgo
    if riesgos_altos or riesgos_medios:
        
        # Determinar tipo global
        if len(riesgos_altos) >= 3:
            tipo = "CRÍTICO"
        elif len(riesgos_altos) >= 1:
            tipo = "ALTO"
        elif len(riesgos_medios) >= 2:
            tipo = "MEDIO"
        else:
            tipo = "BAJO-MEDIO"
        
        print(f"\nEstudiante: {row['Estudiante']}")
        print(f"Nota Final: {row['Nota_Final']:.2f}")
        print(f"Tipo de Riesgo: {tipo}")
        
        if riesgos_altos:
            print(f"Riesgo ALTO en: {', '.join(riesgos_altos)}")
        
        if riesgos_medios:
            print(f"Riesgo MEDIO en: {', '.join(riesgos_medios)}")
        
        print("-"*50)

# =============================================================================
# MODELO MACHINE LEARNING - RANDOM FOREST REGRESSOR
# =============================================================================

print("\n" + "="*70)
print("MODELO RANDOM FOREST - PREDICCIÓN DE NOTA FINAL")
print("="*70)

# Variables de entrada
X = df[['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3']]

# Variable objetivo
y = df['Nota_Final']

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# Entrenamiento
rf_model.fit(X_train, y_train)

# Predicción en test
y_pred = rf_model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (Error medio absoluto): {mae:.3f}")
print(f"R2 Score: {r2:.3f}")

# =============================================================================
# PREDICCIONES AJUSTADAS (CON MATERIA)
# =============================================================================

df['Prediccion_RF'] = rf_model.predict(X)

df['Prediccion_Ajustada'] = 0.5 * df['Prediccion_RF'] + 0.5 * df['Nota_Final']

df['Riesgo_RF'] = df['Prediccion_Ajustada'].apply(
    lambda x: 'ALTO' if x < 3 
    else ('MEDIO' if x < 3.6 else 'BAJO')
)

print("\n" + "="*70)
print("PREDICCIONES CON RANDOM FOREST (AJUSTADAS)")
print("="*70)

print(df[['Estudiante', 'Materia', 'Nota_Final', 'Prediccion_Ajustada', 'Riesgo_RF']].to_string(index=False))


# =============================================================================
# COMPARACIÓN
# =============================================================================

print("\n" + "="*70)
print("COMPARACIÓN MODELOS (PROBABILIDAD vs RANDOM FOREST)")
print("="*70)

comparacion = df[['Estudiante', 'Materia', 'Nota_Final', 'Prediccion_RF', 'Riesgo_RF', 'Prob_Nota_Final']]

print(comparacion.to_string(index=False))


# =============================================================================
# INCONSISTENCIAS
# =============================================================================

df['Prob_num'] = df['Prob_Nota_Final'].str.replace('%','').astype(float)/100

inconsistentes = df[
    ((df['Prob_num'] < 0.5) & (df['Prediccion_RF'] >= 3)) |
    ((df['Prob_num'] >= 0.5) & (df['Prediccion_RF'] < 3))
]

print("\n" + "="*70)
print("CASOS INCONSISTENTES (REVISIÓN DOCENTE)")
print("="*70)

if not inconsistentes.empty:
    print(inconsistentes[['Estudiante','Materia','Nota_Final','Prediccion_RF','Prob_Nota_Final']])
else:
    print("No hay inconsistencias relevantes")


def predecir_estudiante(n1, n2, n3, materia="NO ESPECIFICADA"):

    entrada = pd.DataFrame([[n1, n2, n3]], 
                           columns=['Nota_Corte1', 'Nota_Corte2', 'Nota_Corte3'])
    
    pred_rf = rf_model.predict(entrada)[0]
    
    pred_real = 0.5 * pred_rf + 0.5 * (0.3*n1 + 0.3*n2 + 0.4*n3)
    
    prob = calculate_probability(pred_real, nota_final_std)
    
    if pred_real < 3:
        riesgo = "ALTO"
    elif pred_real < 3.6:
        riesgo = "MEDIO"
    else:
        riesgo = "BAJO"
    
    print("\n" + "="*60)
    print("PREDICCIÓN PARA ESTUDIANTE (CON MATERIA)")
    print("="*60)
    print(f"Materia: {materia}")
    print(f"Notas: Corte1={n1}, Corte2={n2}, Corte3={n3}")
    print(f"Predicción RF: {pred_rf:.2f}")
    print(f"Predicción Ajustada: {pred_real:.2f}")
    print(f"Probabilidad: {prob:.2%}")
    print(f"Nivel de riesgo: {riesgo}")
    print("="*60)
    
    return pred_real, prob, riesgo


# =============================================================================
# PRUEBAS
# =============================================================================

predecir_estudiante(2.5, 2.0, 2.2, "ALGORITMOS")
predecir_estudiante(3.0, 3.2, 3.5, "BASES DE DATOS")
predecir_estudiante(3.0, 3.1, 3.2, "POO II")
predecir_estudiante(4.5, 4.2, 4.8, "INGENIERÍA DE SOFTWARE")

# =============================================================================
# 📄 REPORTE EXPLICADO COMPLETO
# =============================================================================

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

pdf_path = f"{output_dir}/reporte_explicado.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
styles = getSampleStyleSheet()

content = []

# =============================================================================
# HEADER (LOGO ARRIBA DERECHA + FOOTER)
# =============================================================================
def draw_header(canvas, doc):
    logo_path = "Fup.jpeg"

    # LOGO arriba derecha
    if os.path.exists(logo_path):
        canvas.drawImage(
            logo_path,
            doc.pagesize[0] - 130,   # derecha
            doc.pagesize[1] - 70,    # arriba
            width=100,
            height=50,
            preserveAspectRatio=True,
            mask='auto'
        )

    # FOOTER institucional
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        40,
        20,
        f"Fundación Universitaria de Popayán — {datetime.now().strftime('%d/%m/%Y')}"
    )

# =============================================================================
# PORTADA
# =============================================================================
content.append(Paragraph("REPORTE ANALÍTICO ACADÉMICO EXPLICADO", styles['Title']))
content.append(Spacer(1, 20))

content.append(Paragraph(
    "Este reporte presenta un análisis completo del desempeño académico "
    "de los estudiantes, integrando estadística, probabilidad y modelos "
    "de machine learning para identificar niveles de riesgo.",
    styles['Normal']
))

content.append(PageBreak())


# =============================================================================
# 1. DESCRIPCIÓN DEL MODELO
# =============================================================================
content.append(Paragraph("1. Metodología del Análisis", styles['Heading1']))

texto_modelo = """
El análisis combina dos enfoques:

1. Modelo probabilístico:
Se calcula la probabilidad de aprobar usando una distribución normal,
considerando como umbral la nota mínima de aprobación (3.0).

2. Modelo de Machine Learning (Random Forest):
Se entrena un modelo supervisado que aprende patrones entre los cortes
y la nota final.

3. Modelo híbrido:
Se combina la predicción del modelo con la nota real, logrando mayor estabilidad.

Este enfoque permite no solo predecir, sino interpretar el riesgo académico.
"""

content.append(Paragraph(texto_modelo, styles['Normal']))
content.append(Spacer(1, 20))


# =============================================================================
# 2. DISTRIBUCIÓN DE NOTAS
# =============================================================================
content.append(Paragraph("2. Distribución de Notas", styles['Heading1']))

img1 = f"{output_dir}/1_distribucion_notas.png"

texto1 = """
Este gráfico muestra la distribución de las notas en cada corte y la nota final.

Interpretación:
- Se observa la dispersión de las notas.
- La línea roja indica el umbral de aprobación (3.0).
- Valores por debajo de esta línea representan riesgo académico.

Conclusión:
Permite identificar si el grupo tiene tendencia al bajo o alto desempeño.
"""

content.append(Paragraph(texto1, styles['Normal']))

try:
    content.append(Image(img1, width=5*inch, height=3*inch))
except:
    content.append(Paragraph("Gráfica no disponible.", styles['Normal']))

content.append(PageBreak())


# =============================================================================
# 3. PROBABILIDAD DE APROBACIÓN
# =============================================================================
content.append(Paragraph("3. Probabilidad de Aprobación", styles['Heading1']))

img2 = f"{output_dir}/2_probabilidad_por_corte.png"

texto2 = """
Este gráfico representa la probabilidad de aprobar en cada corte.

Interpretación:
- Valores cercanos a 1 indican alta probabilidad de aprobar.
- Valores cercanos a 0 indican alto riesgo.
- La línea roja (0.5) separa aprobado vs riesgo.

Conclusión:
Permite identificar en qué corte los estudiantes presentan mayor dificultad.
"""

content.append(Paragraph(texto2, styles['Normal']))

try:
    content.append(Image(img2, width=5*inch, height=3*inch))
except:
    content.append(Paragraph("Gráfica no disponible.", styles['Normal']))

content.append(PageBreak())


# =============================================================================
# 4. RELACIÓN NOTA vs PROBABILIDAD
# =============================================================================
content.append(Paragraph("4. Relación entre Nota y Probabilidad", styles['Heading1']))

img3 = f"{output_dir}/3_relacion_notas_probabilidad.png"

texto3 = """
Este gráfico muestra la relación entre la nota obtenida y la probabilidad de aprobar.

Interpretación:
- Existe una relación creciente entre nota y probabilidad.
- La zona crítica está cerca de la nota 3.0.
- La curva suavizada muestra el comportamiento real del sistema.

Conclusión:
Confirma que pequeñas variaciones cerca del umbral cambian significativamente el riesgo.
"""

content.append(Paragraph(texto3, styles['Normal']))

try:
    content.append(Image(img3, width=5*inch, height=3*inch))
except:
    content.append(Paragraph("Gráfica no disponible.", styles['Normal']))

content.append(PageBreak())


# =============================================================================
# 5. MODELO DE DISTRIBUCIÓN NORMAL
# =============================================================================
content.append(Paragraph("5. Modelo Probabilístico", styles['Heading1']))

img4 = f"{output_dir}/4_distribucion_normal_cortes.png"

texto4 = """
Este gráfico muestra las distribuciones normales de cada corte.

Interpretación:
- Cada curva representa cómo se distribuyen las notas.
- El área sombreada indica probabilidad de aprobación.
- Mayor desplazamiento a la derecha = mejor desempeño.

Conclusión:
Permite modelar matemáticamente el comportamiento académico.
"""

content.append(Paragraph(texto4, styles['Normal']))

try:
    content.append(Image(img4, width=5*inch, height=3*inch))
except:
    content.append(Paragraph("Gráfica no disponible.", styles['Normal']))

content.append(PageBreak())


# =============================================================================
# 6. MATRIZ DE CORRELACIÓN
# =============================================================================
content.append(Paragraph("6. Correlación entre Cortes", styles['Heading1']))

img6 = f"{output_dir}/6_correlacion_cortes.png"

texto6 = """
Este análisis muestra la relación entre los diferentes cortes.

Interpretación:
- Valores cercanos a 1 indican fuerte relación.
- Permite identificar qué corte influye más en la nota final.

Conclusión:
Ayuda a entender la estructura del aprendizaje del estudiante.
"""

content.append(Paragraph(texto6, styles['Normal']))

try:
    content.append(Image(img6, width=5*inch, height=3*inch))
except:
    content.append(Paragraph("Gráfica no disponible.", styles['Normal']))

content.append(PageBreak())


# =============================================================================
# 7. RESULTADOS DEL MODELO
# =============================================================================
content.append(Paragraph("7. Resultados del Modelo", styles['Heading1']))

texto7 = f"""
Métricas del modelo:

• Error medio absoluto (MAE): {mae:.3f}
• Coeficiente de determinación (R²): {r2:.3f}

Interpretación:
- MAE bajo indica buena precisión.
- R² cercano a 1 indica alta capacidad predictiva.

Conclusión:
El modelo es adecuado para predicción académica.
"""

content.append(Paragraph(texto7, styles['Normal']))
content.append(Spacer(1, 20))


# =============================================================================
# 8. ANÁLISIS DE RIESGO
# =============================================================================
content.append(Paragraph("8. Clasificación de Riesgo", styles['Heading1']))

texto8 = """
Los estudiantes se clasifican en:

• ALTO: nota < 3.0 → riesgo de perder
• MEDIO: 3.0 - 3.6 → zona crítica
• BAJO: > 3.6 → alta probabilidad de aprobar

Este sistema permite priorizar intervenciones académicas.
"""

content.append(Paragraph(texto8, styles['Normal']))
content.append(Spacer(1, 20))


# =============================================================================
# 9. CONCLUSIONES
# =============================================================================
content.append(Paragraph("9. Conclusiones", styles['Heading1']))

texto9 = """
• El modelo híbrido mejora la estabilidad de predicción
• Se identifican claramente estudiantes en riesgo
• El análisis por cortes permite intervención temprana
• Las materias pueden ser evaluadas estratégicamente

Este enfoque puede ser utilizado como sistema institucional
de alerta temprana académica.
"""

content.append(Paragraph(texto9, styles['Normal']))

# =============================================================================
# 🔥 DATOS DE CONSOLA AL PDF
# =============================================================================

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from datetime import datetime

# -----------------------------------------------------------------------------
# TABLA COMPLETA DE RESULTADOS (SIN PÉRDIDA DE FILAS)
# -----------------------------------------------------------------------------
from reportlab.platypus import PageBreak, Table, TableStyle

content.append(PageBreak())
content.append(Paragraph("10. Resultados Detallados", styles['Heading1']))

# Encabezado
header = ['Estudiante', 'Materia', 'Nota Final', 'Predicción', 'Riesgo']

# Datos (sin encabezado)
data_rows = []
for _, row in df.iterrows():
    data_rows.append([
        row['Estudiante'][:20],
        row['Materia'][:20],
        f"{row['Nota_Final']:.2f}",
        f"{row['Prediccion_Ajustada']:.2f}",
        row['Riesgo_RF']
    ])

# 🔥 Tamaño real del bloque (sin contar header)
chunk_size = 24  

for i in range(0, len(data_rows), chunk_size):
    
    bloque = [header] + data_rows[i:i+chunk_size]  # 👈 SIEMPRE agrega header
    
    t = Table(bloque, repeatRows=1)
    
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTSIZE', (0,0), (-1,-1), 7)
    ]))
    
    content.append(t)
    
    # 🔥 SOLO agregar salto si NO es el último bloque
    if i + chunk_size < len(data_rows):
        content.append(PageBreak())


# -----------------------------------------------------------------------------
# ESTUDIANTES EN RIESGO (CONSOLA / PDF)
# -----------------------------------------------------------------------------
content.append(Spacer(1, 20))
content.append(Paragraph("11. Estudiantes en Riesgo", styles['Heading1']))

# 🔥 AJUSTE AQUÍ (ordenados de peor a mejor)
riesgo_df = df[df['Riesgo_RF'] != 'BAJO'].sort_values(
    by='Prediccion_Ajustada', ascending=True
)

if not riesgo_df.empty:
    for _, row in riesgo_df.iterrows():
        texto = f"""
        <b>Código:</b> {row['Codigo']}<br/>
        <b>Estudiante:</b> {row['Estudiante']}<br/>
        <b>Materia:</b> {row['Materia']}<br/>
        <b>Nota Final:</b> {row['Nota_Final']:.2f}<br/>
        <b>Predicción:</b> {row['Prediccion_Ajustada']:.2f}<br/>
        <b>Riesgo:</b> {row['Riesgo_RF']}<br/>
        <br/>
        """
        content.append(Paragraph(texto, styles['Normal']))
        content.append(Spacer(1, 10))  # separación visual

else:
    content.append(Paragraph("No hay estudiantes en riesgo.", styles['Normal']))


# -----------------------------------------------------------------------------
# CASOS INCONSISTENTES
# -----------------------------------------------------------------------------
content.append(Spacer(1, 20))
content.append(Paragraph("12. Casos Inconsistentes", styles['Heading1']))

if not inconsistentes.empty:
    for _, row in inconsistentes.iterrows():
        texto = f"""
        • {row['Estudiante']}<br/>
        Materia: {row['Materia']}<br/>
        Nota Final: {row['Nota_Final']:.2f}<br/>
        Predicción RF: {row['Prediccion_RF']:.2f}<br/>
        Probabilidad: {row['Prob_Nota_Final']}<br/>
        """
        content.append(Paragraph(texto, styles['Normal']))
else:
    content.append(Paragraph("No se detectaron inconsistencias.", styles['Normal']))


# -----------------------------------------------------------------------------
# RESUMEN FINAL (TIPO CONSOLA)
# -----------------------------------------------------------------------------
content.append(Spacer(1, 20))
content.append(Paragraph("13. Resumen Final", styles['Heading1']))

resumen_final = f"""
Total estudiantes: {len(df)}<br/>
Promedio final: {df['Nota_Final'].mean():.2f}<br/>
Estudiantes en riesgo: {len(riesgo_df)}<br/>
% aprobación: {df['Objetivo_Cumplido'].mean():.2%}
"""

content.append(Paragraph(resumen_final, styles['Normal']))

# =============================================================================
# GENERAR PDF
# =============================================================================
doc.build(content, onFirstPage=draw_header, onLaterPages=draw_header)

print(f"\n📄 REPORTE EXPLICADO generado en: {pdf_path}")
