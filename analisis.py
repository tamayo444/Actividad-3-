import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATOS DEL PROBLEMA
# ==========================================
L = 2000.0       # Longitud [mm]
E = 200000.0     # Módulo de elasticidad [N/mm^2]
A = 1000.0       # Área [mm^2]
P = 20000.0      # Carga puntual en +x [N]
q = 10.0         # Carga distribuida uniforme en +x [N/mm]
EA = E * A       # Rigidez axial [N]

# ==========================================
# 2. FUNCIONES DE REFERENCIA Y ERROR
# ==========================================
def solucion_analitica(x):
    """Calcula el desplazamiento analitico exacto u(x) [mm]."""
    return (1.0 / EA) * ((P + q * L) * x - 0.5 * q * (x**2))

def calcular_error_puntos_medios(n, u_mef, x_nodos):
    """Calcula el error absoluto maximo evaluado en los puntos medios."""
    errores = []
    for i in range(n):
        x_mid = 0.5 * (x_nodos[i] + x_nodos[i+1])
        u_exacto_mid = solucion_analitica(x_mid)
        u_mef_mid = 0.5 * (u_mef[i] + u_mef[i+1]) # Interpolacion lineal MEF
        errores.append(abs(u_exacto_mid - u_mef_mid))
    return max(errores)

# ==========================================
# 3. SOLVER MEF (Rigidez, Ensamblaje y Resolución)
# ==========================================
def resolver_mef_barra(n, fijar_extremo_L=False):
    """
    Resuelve el problema de la barra axial para 'n' elementos.
    Si fijar_extremo_L=True, impone u(L) = 0.
    """
    h = L / n
    x_nodos = np.linspace(0, L, n + 1)
    k_elem = (EA / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    
    # Matriz Global K y Vector de Fuerzas Global f
    K = np.zeros((n + 1, n + 1))
    f = np.zeros(n + 1)
    
    # Ensamblaje
    for i in range(n):
        K[i:i+2, i:i+2] += k_elem
        # Carga distribuida equivalente en los nodos
        f[i] += q * h / 2.0
        f[i+1] += q * h / 2.0
        
    # Carga puntual en el extremo
    f[-1] += P
    
    # Identificación de GDLS (Grados de Libertad)
    gdl_restringidos = [0]
    if fijar_extremo_L:
        gdl_restringidos.append(n)
        
    gdl_libres = [i for i in range(n + 1) if i not in gdl_restringidos]
    
    # Aplicar Condiciones de Borde (Reducción del Sistema)
    K_libre = K[np.ix_(gdl_libres, gdl_libres)]
    f_libre = f[gdl_libres]
    
    # Resolver desplazamientos en GDLS libres
    u_libre = np.linalg.solve(K_libre, f_libre)
    
    # Reconstruir vector de desplazamientos global
    u = np.zeros(n + 1)
    u[gdl_libres] = u_libre
    
    # Recuperación de Reacciones globales R = K*u - f
    R = K @ u - f
    
    return x_nodos, u, R, K, f

# ==========================================
# 4. EJECUCIÓN DE LAS MALLAS Y RESULTADOS
# ==========================================
mallas = [2, 4, 8]
errores_puntos_medios = []

print("="*65)
print(f"{'n':<4} | {'u(L) [mm]':<10} | {'R_0 [N]':<10} | {'Error Max P.Medio [mm]':<22} | {'Balance'}")
print("="*65)

for n in mallas:
    x_nodos, u, R, K, f = resolver_mef_barra(n)
    
    u_L = u[-1]
    R_0 = R[0]
    
    # Verificación de Control de Equilibrio: |R_0 + P + qL| / |P + qL| <= 1e-8
    balance = abs(R_0 + P + q * L) / abs(P + q * L)
    
    # Error en puntos medios
    err_mid = calcular_error_puntos_medios(n, u, x_nodos)
    errores_puntos_medios.append(err_mid)
    
    print(f"{n:<4} | {u_L:<10.4f} | {R_0:<10.1f} | {err_mid:<22.8f} | {balance:.2e}")

print("="*65)

# ==========================================
# 5. DIAGNÓSTICO: CAMBIO DE CONTORNO (u(L) = 0) CON n=8
# ==========================================
print("\n--- DIAGNÓSTICO: RESTRICCIÓN ADICIONAL u(L) = 0 (Malla n=8) ---")
x_nodos, u_alt, R_alt, _, _ = resolver_mef_barra(n=8, fijar_extremo_L=True)

print(f"Desplazamiento u(L) : {u_alt[-1]:.4f} mm")
print(f"Reacción en x=0 (R_0): {R_alt[0]:.1f} N")
print(f"Reacción en x=L (R_L): {R_alt[-1]:.1f} N")

balance_alt = abs(R_alt[0] + R_alt[-1] + P + q * L) / abs(P + q * L)
print(f"Balance de Equilibrio Global: {balance_alt:.2e} (Cumple Cierre Algebraico)")

# ==========================================
# 6. GENERACIÓN Y GUARDADO DEL GRÁFICO
# ==========================================
os.makedirs('informe/figures', exist_ok=True)

plt.figure(figsize=(6, 4), dpi=300)
plt.plot(mallas, errores_puntos_medios, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Error máx. punto medio')
plt.yscale('log')
plt.xlabel('Número de elementos ($n$)', fontsize=11)
plt.ylabel('Error absoluto máximo [mm]', fontsize=11)
plt.title('Convergencia del Error en Puntos Medios', fontsize=12, fontweight='bold')
plt.xticks(mallas, labels=['2', '4', '8'])
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

ruta_grafico = 'informe/figures/error_mesh.png'
plt.savefig(ruta_grafico)
print(f"\n[OK] Gráfico generado y guardado exitosamente en: {ruta_grafico}")
