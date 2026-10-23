"""
=============================================================================
  AgriRob-SN : Modèle cinématique et dynamique d'un robot manipulateur 6 ddl
  pour applications agricoles au Sénégal (arachide, mil, maraîchage)
=============================================================================
  Convention : Denavit-Hartenberg modifiée (Khalil-Kleinfinger)
  Formalisme dynamique : Lagrange
  Auteur : (à compléter)
=============================================================================
"""
import numpy as np
from numpy import cos, sin, pi

# =============================================================================
# 1. PARAMÈTRES PHYSIQUES DU ROBOT
# =============================================================================

# Paramètres de Denavit-Hartenberg modifiés
# Format : [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
# Robot 6 ddl type RRR-RRR (épaule + poignet sphérique)
DH_PARAMS = np.array([
    [0.0,        0.0,    0.15,   0.0],   # Joint 1 (rotation base)
    [-pi/2,      0.0,    0.0,    0.0],   # Joint 2 (épaule)
    [0.0,        0.40,   0.0,    0.0],   # Joint 3 (coude)
    [-pi/2,      0.05,   0.40,   0.0],   # Joint 4 (poignet roll)
    [pi/2,       0.0,    0.0,    0.0],   # Joint 5 (poignet pitch)
    [-pi/2,      0.0,    0.10,   0.0],   # Joint 6 (poignet yaw)
])

# Masses des segments (kg) - bras léger en aluminium
LINK_MASSES = np.array([2.5, 3.0, 2.5, 1.5, 1.0, 1.5])

# Position des centres de masse dans le repère de chaque segment (m)
# [x, y, z] dans le repère i
LINK_COM = np.array([
    [0.0,   0.0,   0.075],
    [0.20,  0.0,   0.0],
    [0.20,  0.0,   0.0],
    [0.0,   0.0,   0.20],
    [0.0,   0.0,   0.0],
    [0.0,   0.0,   0.05],
])

# Tenseurs d'inertie diagonaux (Ixx, Iyy, Izz) en kg.m^2 - estimation
LINK_INERTIAS = np.array([
    [0.012, 0.012, 0.008],
    [0.040, 0.040, 0.010],
    [0.030, 0.030, 0.008],
    [0.005, 0.005, 0.003],
    [0.002, 0.002, 0.002],
    [0.002, 0.002, 0.001],
])

# Limites articulaires (rad)
JOINT_LIMITS = np.array([
    [-pi,    pi   ],
    [-pi/2,  pi/2 ],
    [-2.5,   2.5  ],
    [-pi,    pi   ],
    [-pi/2,  pi/2 ],
    [-pi,    pi   ],
])

# Constantes physiques
GRAVITY = 9.81  # m/s^2

# =============================================================================
# 2. MODÈLE GÉOMÉTRIQUE DIRECT (MGD)
# =============================================================================

def dh_transform(alpha, a, d, theta):
    """
    Matrice de transformation homogène selon DH modifiée (Khalil-Kleinfinger)
    
    Args:
        alpha : angle de torsion (rad)
        a     : longueur du segment (m)
        d     : décalage (m)
        theta : angle articulaire (rad)
    Returns:
        T : matrice 4x4 homogène
    """
    ca, sa = cos(alpha), sin(alpha)
    ct, st = cos(theta), sin(theta)
    T = np.array([
        [ct,         -st,        0.0,    a       ],
        [st*ca,      ct*ca,      -sa,    -sa*d   ],
        [st*sa,      ct*sa,      ca,     ca*d    ],
        [0.0,        0.0,        0.0,    1.0     ],
    ])
    return T


def forward_kinematics(q, return_all=False):
    """
    Modèle géométrique direct : calcule la pose de l'effecteur 0_T_6
    
    Args:
        q : vecteur articulaire (6,)
        return_all : si True, retourne aussi tous les 0_T_i intermédiaires
    Returns:
        T_06 : matrice 4x4 de pose de l'effecteur
        (transforms : liste de toutes les 0_T_i si return_all=True)
    """
    T = np.eye(4)
    transforms = [T.copy()]
    for i in range(6):
        alpha, a, d, th_off = DH_PARAMS[i]
        Ti = dh_transform(alpha, a, d, q[i] + th_off)
        T = T @ Ti
        transforms.append(T.copy())
    if return_all:
        return T, transforms
    return T


# =============================================================================
# 3. JACOBIENNE GÉOMÉTRIQUE
# =============================================================================

def jacobian(q):
    """
    Jacobienne géométrique 6x6 : [v; omega]_e = J(q) * q_dot
    Convention DH modifiée : l'axe de l'articulation i est l'axe z_i du repère i
    (après la transformation 0_T_i).
    
    Args:
        q : vecteur articulaire (6,)
    Returns:
        J : matrice 6x6 (3 lignes vitesse linéaire, 3 lignes vitesse angulaire)
    """
    _, transforms = forward_kinematics(q, return_all=True)
    p_e = transforms[-1][:3, 3]   # position de l'effecteur
    J = np.zeros((6, 6))
    for i in range(6):
        # En DH modifiée, l'articulation i est portée par l'axe z du repère i
        # (transforms[i+1] = 0_T_i)
        T_i = transforms[i+1]
        z_i = T_i[:3, 2]
        p_i = T_i[:3, 3]
        # Articulation rotoïde
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J


# =============================================================================
# 4. MODÈLE GÉOMÉTRIQUE INVERSE (MCI) - méthode numérique
# =============================================================================

def inverse_kinematics(target_pose, q_init, max_iter=200, tol=1e-4, lambda_dls=0.01,
                       position_only=False):
    """
    MCI par méthode des moindres carrés amortis (Damped Least Squares).
    Robuste près des singularités.
    
    Args:
        target_pose : matrice 4x4 cible
        q_init      : configuration initiale (6,)
        max_iter    : nombre maximum d'itérations
        tol         : tolérance sur l'erreur
        lambda_dls  : facteur d'amortissement
        position_only : si True, n'asservit que la position (3 DDL libres)
    Returns:
        q : configuration articulaire solution
        converged : booléen
        history : historique des erreurs
    """
    q = q_init.copy().astype(float)
    history = []
    target_p = target_pose[:3, 3]
    target_R = target_pose[:3, :3]
    
    for it in range(max_iter):
        T_curr = forward_kinematics(q)
        curr_p = T_curr[:3, 3]
        curr_R = T_curr[:3, :3]
        
        # Erreur de position
        err_p = target_p - curr_p
        
        if position_only:
            err = err_p
            J_full = jacobian(q)
            J = J_full[:3, :]   # 3x6
        else:
            # Erreur d'orientation par formulation log(R_target * R_curr^T)
            R_err = target_R @ curr_R.T
            cos_th = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
            theta = np.arccos(cos_th)
            if abs(theta) < 1e-8:
                err_o = np.zeros(3)
            else:
                err_o = (theta / (2*sin(theta))) * np.array([
                    R_err[2,1] - R_err[1,2],
                    R_err[0,2] - R_err[2,0],
                    R_err[1,0] - R_err[0,1],
                ])
            err = np.concatenate([err_p, err_o])
            J = jacobian(q)
        
        history.append(np.linalg.norm(err))
        
        if np.linalg.norm(err) < tol:
            return q, True, history
        
        # DLS : delta_q = J^T (J J^T + lambda^2 I)^-1 err
        m = J.shape[0]
        JJt = J @ J.T
        delta_q = J.T @ np.linalg.solve(JJt + (lambda_dls**2) * np.eye(m), err)
        q = q + delta_q
        # Application des limites articulaires
        q = np.clip(q, JOINT_LIMITS[:,0], JOINT_LIMITS[:,1])
    
    return q, False, history


# =============================================================================
# 5. MODÈLE DYNAMIQUE (formalisme de Lagrange)
# =============================================================================

def mass_matrix(q):
    """
    Matrice d'inertie M(q) calculée par la méthode des Jacobiens.
    M(q) = sum_i [ m_i * J_v_i^T J_v_i + J_w_i^T (R_i I_i R_i^T) J_w_i ]
    """
    M = np.zeros((6, 6))
    _, transforms = forward_kinematics(q, return_all=True)
    
    for i in range(6):
        T_i = transforms[i+1]   # 0_T_i (repère du segment i)
        R_i = T_i[:3, :3]
        # Position du centre de masse du segment i dans le repère 0
        p_ci = T_i[:3, 3] + R_i @ LINK_COM[i]
        
        # Jacobienne du COM du segment i
        # Seules les articulations 0..i contribuent
        J_v = np.zeros((3, 6))
        J_w = np.zeros((3, 6))
        for j in range(i+1):
            T_j = transforms[j+1]   # repère 0_T_j (axe z_j)
            z_j = T_j[:3, 2]
            p_j = T_j[:3, 3]
            J_v[:, j] = np.cross(z_j, p_ci - p_j)
            J_w[:, j] = z_j
        
        # Tenseur d'inertie au centre de masse, exprimé en repère 0
        I_local = np.diag(LINK_INERTIAS[i])
        I_world = R_i @ I_local @ R_i.T
        
        M += LINK_MASSES[i] * (J_v.T @ J_v) + J_w.T @ I_world @ J_w
    
    # Symétrisation pour éviter les erreurs numériques
    M = 0.5 * (M + M.T)
    return M


def coriolis_matrix(q, qd, eps=1e-5):
    """
    Matrice C(q, q_dot) par les symboles de Christoffel calculés numériquement.
    c_{ijk} = 0.5 * ( dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i )
    C_{ij} = sum_k c_{ijk} * qd_k
    """
    n = 6
    M0 = mass_matrix(q)
    # Calcul des dérivées partielles dM/dq_k par différences finies
    dM = np.zeros((n, n, n))
    for k in range(n):
        q_p = q.copy(); q_p[k] += eps
        q_m = q.copy(); q_m[k] -= eps
        dM[:, :, k] = (mass_matrix(q_p) - mass_matrix(q_m)) / (2*eps)
    
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c_ijk = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                C[i, j] += c_ijk * qd[k]
    return C


def gravity_vector(q):
    """
    Vecteur de gravité G(q) par la méthode des Jacobiens.
    G_i = -sum_j m_j * g^T * J_v_j[:, i]
    avec g = [0, 0, -9.81]^T
    """
    g_vec = np.array([0.0, 0.0, -GRAVITY])
    G = np.zeros(6)
    _, transforms = forward_kinematics(q, return_all=True)
    
    for j in range(6):
        T_j = transforms[j+1]
        R_j = T_j[:3, :3]
        p_cj = T_j[:3, 3] + R_j @ LINK_COM[j]
        
        J_v = np.zeros((3, 6))
        for k in range(j+1):
            T_k = transforms[k+1]   # axe z_k
            z_k = T_k[:3, 2]
            p_k = T_k[:3, 3]
            J_v[:, k] = np.cross(z_k, p_cj - p_k)
        
        G += -LINK_MASSES[j] * (J_v.T @ g_vec)
    
    return G


def inverse_dynamics(q, qd, qdd):
    """
    Couples articulaires nécessaires : tau = M(q) qdd + C(q,qd) qd + G(q) + F(qd)
    Avec frottements visqueux + secs simplifiés.
    """
    M = mass_matrix(q)
    C = coriolis_matrix(q, qd)
    G = gravity_vector(q)
    # Frottements
    F_v = 0.5   # coefficient visqueux N.m.s/rad
    F_s = 0.3   # frottement sec N.m
    F = F_v * qd + F_s * np.sign(qd)
    tau = M @ qdd + C @ qd + G + F
    return tau


# =============================================================================
# 6. MODÈLE DE LA PLATEFORME MOBILE (sol sableux sénégalais)
# =============================================================================

class MobileBase:
    """
    Plateforme 4 roues motrices avec modèle de roulement sur sol sableux
    (modèle de Bekker-Wong simplifié).
    """
    def __init__(self):
        self.M = 85.0           # masse à vide (kg)
        self.L = 0.80           # empattement (m)
        self.l = 0.70           # voie (m)
        self.r = 0.15           # rayon des roues (m)
        # Paramètres sol Dior (sol sableux du bassin arachidier)
        self.f_r = 0.18         # coefficient résistance roulement (sol sableux)
        self.k_s = 250e3        # raideur sol (Pa/m^n) - typique sol sableux
        self.n = 1.0            # exposant Bekker
        self.slip_coeff = 0.10  # glissement moyen sur sol sableux
        self.eta_motor = 0.80   # rendement moteurs
    
    def rolling_resistance(self, payload, z_sink=0.02):
        """Force de résistance au roulement totale."""
        W = (self.M + payload) * GRAVITY
        F_basic = W * self.f_r
        F_sink = self.k_s * (z_sink ** self.n) * 0.04   # surface contact ~4cm²
        return F_basic + F_sink
    
    def power_consumption(self, v, payload, slope=0.0):
        """Puissance instantanée pour vitesse v et charge donnée."""
        F_resist = self.rolling_resistance(payload)
        F_slope = (self.M + payload) * GRAVITY * sin(slope)
        F_total = F_resist + F_slope
        # Compensation glissement
        v_eff = v / (1.0 - self.slip_coeff)
        P_mech = F_total * v_eff
        P_elec = P_mech / self.eta_motor
        return P_elec


# =============================================================================
# 7. UTILITAIRES
# =============================================================================

def workspace_sample(n_samples=2000):
    """Échantillonne l'espace de travail atteignable (cloud de points)."""
    np.random.seed(42)
    points = []
    for _ in range(n_samples):
        q = np.random.uniform(JOINT_LIMITS[:,0], JOINT_LIMITS[:,1])
        T = forward_kinematics(q)
        points.append(T[:3, 3])
    return np.array(points)


if __name__ == "__main__":
    # Test rapide
    q_test = np.array([0.1, -0.5, 0.8, 0.2, 0.3, 0.0])
    T = forward_kinematics(q_test)
    print("=== Test du modèle AgriRob-SN ===")
    print(f"\nConfiguration test : q = {q_test}")
    print(f"\nPose effecteur 0_T_6 :\n{np.round(T, 4)}")
    print(f"\nPosition effecteur : {np.round(T[:3, 3], 4)} m")
    
    J = jacobian(q_test)
    print(f"\nDéterminant J·J^T : {np.linalg.det(J @ J.T):.6e}")
    
    M = mass_matrix(q_test)
    print(f"\nMatrice d'inertie M(q) (diag) : {np.round(np.diag(M), 4)}")
    
    G = gravity_vector(q_test)
    print(f"\nVecteur de gravité G(q) (N.m) : {np.round(G, 4)}")
    
    base = MobileBase()
    P = base.power_consumption(v=0.5, payload=10.0)
    print(f"\nPuissance plateforme (v=0.5m/s, 10kg) : {P:.1f} W")
