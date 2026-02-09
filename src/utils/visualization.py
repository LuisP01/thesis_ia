import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib

matplotlib.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def guardar_grafico(fig, cedula: str, tipo: str, nombre_grafico: str):
    base_dir = "graficos_tesis"
    user_dir = os.path.join(base_dir, cedula, tipo)
    os.makedirs(user_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{nombre_grafico}_{timestamp}.png"
    filepath = os.path.join(user_dir, filename)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Gráfico guardado: {filepath}")
    return filepath

def grafico_serie_completa(df, pred, intervalo, proxima_fecha, cedula, tipo, modelo_usado):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    ax[0, 0].plot(df['ds'], df['y'], 'b-', linewidth=2, label='Consumo histórico', marker='o', markersize=4)
    ax[0, 0].axvline(x=df['ds'].iloc[-1], color='gray', linestyle='--', alpha=0.5, label='Último dato')
    
    pred_date = pd.Timestamp(proxima_fecha + "-01")
    ax[0, 0].plot(pred_date, pred, 'ro', markersize=10, label=f'Predicción: {pred:.1f}', zorder=5)
    ax[0, 0].errorbar(pred_date, pred, yerr=[[pred-intervalo[0]], [intervalo[1]-pred]], 
                      fmt='none', ecolor='red', capsize=5, capthick=2, label='Intervalo 95%')
    
    ax[0, 0].set_title(f'Consumo de {tipo.upper()} - Usuario: {cedula}\nModelo: {modelo_usado}', fontsize=14, fontweight='bold')
    ax[0, 0].set_xlabel('Fecha', fontsize=12)
    ax[0, 0].set_ylabel(f'Consumo ({tipo})', fontsize=12)
    ax[0, 0].legend(loc='best')
    ax[0, 0].grid(True, alpha=0.3)
    
    ax[0, 1].plot(df['ds'], df['y'], 'b-', alpha=0.5, label='Datos reales')
    
    x_numeric = np.arange(len(df))
    coef = np.polyfit(x_numeric, df['y'].values, 1)
    trend_line = np.polyval(coef, x_numeric)
    ax[0, 1].plot(df['ds'], trend_line, 'r--', linewidth=2, label=f'Tendencia: y={coef[0]:.2f}x+{coef[1]:.2f}')
    
    ax[0, 1].set_title('Tendencia Lineal', fontsize=12)
    ax[0, 1].set_xlabel('Fecha')
    ax[0, 1].set_ylabel('Consumo')
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)
    
    ax[1, 0].hist(df['y'].values, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    ax[1, 0].axvline(df['y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["y"].mean():.1f}')
    ax[1, 0].axvline(df['y'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["y"].median():.1f}')
    
    ax[1, 0].set_title('Distribución de Consumo Histórico', fontsize=12)
    ax[1, 0].set_xlabel('Consumo')
    ax[1, 0].set_ylabel('Frecuencia')
    ax[1, 0].legend()
    
    ax[1, 1].axis('off')
    
    stats_text = f"""
    ESTADÍSTICAS RESUMEN:
    • Período: {df['ds'].iloc[0].strftime('%b %Y')} - {df['ds'].iloc[-1].strftime('%b %Y')}
    • Meses totales: {len(df)}
    • Consumo promedio: {df['y'].mean():.1f}
    • Consumo mínimo: {df['y'].min():.1f}
    • Consumo máximo: {df['y'].max():.1f}
    • Desviación estándar: {df['y'].std():.1f}
    
    PREDICCIÓN:
    • Próximo mes: {proxima_fecha}
    • Valor predicho: {pred:.1f}
    • Intervalo: [{intervalo[0]:.1f}, {intervalo[1]:.1f}]
    • Modelo utilizado: {modelo_usado}
    """
    
    ax[1, 1].text(0.1, 0.95, stats_text, transform=ax[1, 1].transAxes, 
                 fontsize=11, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    ruta = guardar_grafico(fig, cedula, tipo, "serie_completa")
    return ruta

def grafico_comparacion_modelos(resultados_lineal, resultados_prophet, cedula, tipo):
    if not resultados_prophet:
        print("No hay resultados Prophet para comparar")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    modelos = ['Regresión Lineal']
    mape_valores = [resultados_lineal[0]] if resultados_lineal else [0]
    
    if resultados_prophet:
        modelos.append('Prophet (Mejor)')
        mape_valores.append(resultados_prophet[0]['mape'])
    
    bars = ax1.bar(modelos, mape_valores, color=['skyblue', 'lightcoral'])
    
    for bar, valor in zip(bars, mape_valores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{valor:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Comparación de MAPE entre Modelos', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAPE (%)')
    ax1.set_xlabel('Modelo')
    ax1.grid(axis='y', alpha=0.3)
    
    ax1.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Excelente (<10%)')
    ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Aceptable (<20%)')
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Revisar (>30%)')
    ax1.legend(loc='upper right')
    
    if len(resultados_prophet) > 1:
        cps_vals = [r['cps'] for r in resultados_prophet[:5]]
        sps_vals = [r['sps'] for r in resultados_prophet[:5]]
        mape_vals = [r['mape'] for r in resultados_prophet[:5]]
        
        indices = np.arange(len(cps_vals))
        width = 0.25
        
        bars1 = ax2.bar(indices - width, cps_vals, width, label='CPS', alpha=0.7)
        bars2 = ax2.bar(indices, sps_vals, width, label='SPS', alpha=0.7)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(indices, mape_vals, 'ko-', linewidth=2, markersize=8, label='MAPE')
        ax2_twin.set_ylabel('MAPE (%)', color='black')
        ax2_twin.tick_params(axis='y', labelcolor='black')
        
        ax2.set_title('Variación de Hiperparámetros Prophet', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Configuración (Top 5)')
        ax2.set_ylabel('Valor del Parámetro')
        ax2.set_xticks(indices)
        ax2.set_xticklabels([f'Conf {i+1}' for i in indices])
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
    
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Solo una configuración Prophet evaluada', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Análisis Comparativo de Modelos - {cedula} ({tipo.upper()})', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    ruta = guardar_grafico(fig, cedula, tipo, "comparacion_modelos")
    return ruta

def grafico_prediccion_detallada(df, reales, predicciones, cedula, tipo, modelo_tipo):
    if not reales or not predicciones:
        print("No hay datos de validación para gráfico detallado")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fechas_pred = df['ds'].iloc[-len(reales):].reset_index(drop=True)
    
    ax.plot(fechas_pred, reales, 'b-', linewidth=2, marker='o', markersize=6, 
            label='Valores Reales', alpha=0.8)
    ax.plot(fechas_pred, predicciones, 'r--', linewidth=2, marker='s', markersize=6,
            label='Predicciones del Modelo', alpha=0.8)
    
    for fecha, real, pred in zip(fechas_pred, reales, predicciones):
        ax.plot([fecha, fecha], [real, pred], 'gray', alpha=0.3, linewidth=1)
    
    errores = [abs(r-p) for r, p in zip(reales, predicciones)]
    mape = np.mean([abs(r-p)/r*100 for r, p in zip(reales, predicciones) if r > 0])
    
    ax.fill_between(fechas_pred, 
                   [p - np.mean(errores) for p in predicciones],
                   [p + np.mean(errores) for p in predicciones],
                   alpha=0.2, color='red', label='Margen de error promedio')
    
    ax.set_title(f'Validación Walk-Forward - Modelo {modelo_tipo.upper()}\nMAPE: {mape:.2f}%', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(f'Consumo ({tipo})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    stats_text = f"""
    Estadísticas de Validación:
    • Meses evaluados: {len(reales)}
    • Error absoluto medio: {np.mean(errores):.2f}
    • Error máximo: {np.max(errores):.2f}
    • Error mínimo: {np.min(errores):.2f}
    """
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    ruta = guardar_grafico(fig, cedula, tipo, f"validacion_{modelo_tipo}")
    return ruta

def grafico_resumen_ejecucion(cedula, tipo, resultados, rutas_graficos):
    fig = plt.figure(figsize=(10, 8))
    
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  
    ax2 = fig.add_subplot(gs[1, 0])  
    ax3 = fig.add_subplot(gs[1, 1])  
    ax4 = fig.add_subplot(gs[2, :])  
    
    ax1.axis('off')
    titulo = f"REPORTE DE EJECUCIÓN - SISTEMA DE PREDICCIÓN\nUsuario: {cedula} | Servicio: {tipo.upper()}"
    ax1.text(0.5, 0.8, titulo, ha='center', va='center', 
            fontsize=16, fontweight='bold', transform=ax1.transAxes)
    
    fecha_ejec = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ax1.text(0.5, 0.6, f"Fecha de ejecución: {fecha_ejec}", 
            ha='center', va='center', fontsize=12, transform=ax1.transAxes)
    
    ax2.axis('off')
    if 'mape_lineal' in resultados and 'mape_prophet' in resultados:
        metricas_text = f"""
        MÉTRICAS DE ERROR:
        
        Regresión Lineal:
        • MAPE: {resultados['mape_lineal']:.2f}%
        
        Prophet:
        • MAPE: {resultados.get('mape_prophet', 'N/A')}
        
        Modelo Seleccionado:
        • {resultados.get('modelo_seleccionado', 'N/A')}
        • Diferencia: {resultados.get('diferencia_mape', 0):.2f}%
        """
    else:
        metricas_text = "Datos de métricas no disponibles"
    
    ax2.text(0.1, 0.9, metricas_text, ha='left', va='top', 
            fontsize=10, transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax3.axis('off')
    decision_text = f"""
    JUSTIFICACIÓN DE DECISIÓN:
    
    {'• Serie muestra fuerte tendencia lineal (R² alto)' 
     if resultados.get('r2', 0) > 0.7 else 
     '• Patrón complejo detectado'}
    
    {'• Baja estacionalidad' 
     if resultados.get('estacionalidad', 0) < 0.3 else 
     '• Alta estacionalidad presente'}
    
    {'• Datos insuficientes para Prophet' 
     if resultados.get('n_meses', 0) < 24 else 
     '• Suficientes datos para modelos complejos'}
    
    RECOMENDACIÓN:
    {resultados.get('recomendacion', 'Revisar manualmente')}
    """
    
    ax3.text(0.1, 0.9, decision_text, ha='left', va='top',
            fontsize=10, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Archivos generados
    ax4.axis('off')
    archivos_text = "ARCHIVOS GENERADOS:\n\n"
    for ruta in rutas_graficos:
        if ruta:
            nombre_archivo = os.path.basename(ruta)
            archivos_text += f"• {nombre_archivo}\n"
    
    ax4.text(0.1, 0.9, archivos_text, ha='left', va='top',
            fontsize=9, transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Resumen Ejecutivo - Documentación Técnica', y=0.98, fontsize=14)
    plt.tight_layout()
    
    ruta = guardar_grafico(fig, cedula, tipo, "resumen_ejecutivo")
    return ruta