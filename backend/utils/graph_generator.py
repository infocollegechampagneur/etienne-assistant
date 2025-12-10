"""Générateur de graphiques mathématiques pour Étienne"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import List, Tuple, Optional
import re
import logging

def detect_graph_request(text: str) -> bool:
    """Détecte si le texte contient une demande de graphique"""
    graph_keywords = [
        'graphique', 'graph', 'courbe', 'tracer', 'diagramme',
        'plot', 'afficher', 'représenter graphiquement', 'dessiner',
        'visualiser', 'fonction', 'équation'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in graph_keywords)

def extract_math_expression(text: str) -> Optional[str]:
    """Extrait une expression mathématique du texte
    
    Exemples:
    - "f(x) = x^2 + 2x + 1"
    - "y = sin(x)"
    - "tracer 2x + 3"
    """
    # Patterns possibles
    patterns = [
        r'[fy]\(x\)\s*=\s*([^,\n]+)',  # f(x) = ... ou y(x) = ...
        r'y\s*=\s*([^,\n]+)',           # y = ...
        r'tracer\s+([^,\n]+)',          # tracer ...
        r'graphique\s+de\s+([^,\n]+)',  # graphique de ...
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def safe_eval_expression(expr: str, x_val: float) -> Optional[float]:
    """Évalue une expression mathématique de manière sécurisée"""
    try:
        # Remplacer les notations courantes
        expr = expr.replace('^', '**')  # Puissance
        expr = expr.replace('x', f'({x_val})')  # Remplacer x par sa valeur
        
        # Créer un namespace sécurisé avec fonctions math
        safe_dict = {
            '__builtins__': {},
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'abs': abs,
            'pi': np.pi,
            'e': np.e
        }
        
        result = eval(expr, safe_dict)
        return float(result)
    except:
        return None

def generate_function_graph(
    expression: str,
    x_range: Tuple[float, float] = (-10, 10),
    title: str = "Graphique de la fonction",
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[str]:
    """Génère un graphique d'une fonction mathématique
    
    Args:
        expression: Expression mathématique (ex: "x**2 + 2*x + 1")
        x_range: Plage de valeurs pour x
        title: Titre du graphique
        figsize: Taille de la figure (largeur, hauteur)
    
    Returns:
        Image encodée en base64 ou None si échec
    """
    try:
        # Générer les valeurs x
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Calculer les valeurs y
        y = []
        for x_val in x:
            y_val = safe_eval_expression(expression, x_val)
            if y_val is None:
                return None
            y.append(y_val)
        
        y = np.array(y)
        
        # Créer le graphique
        plt.figure(figsize=figsize)
        plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {expression}')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        logging.error(f"Erreur génération graphique: {e}")
        return None

def generate_bar_chart(
    categories: List[str],
    values: List[float],
    title: str = "Diagramme à barres",
    xlabel: str = "Catégories",
    ylabel: str = "Valeurs",
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[str]:
    """Génère un diagramme à barres
    
    Returns:
        Image encodée en base64 ou None si échec
    """
    try:
        plt.figure(figsize=figsize)
        plt.bar(categories, values, color='steelblue', alpha=0.8)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        logging.error(f"Erreur génération diagramme: {e}")
        return None

def generate_line_chart(
    x_data: List[float],
    y_data: List[float],
    title: str = "Graphique linéaire",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[str]:
    """Génère un graphique linéaire
    
    Returns:
        Image encodée en base64 ou None si échec
    """
    try:
        plt.figure(figsize=figsize)
        plt.plot(x_data, y_data, 'o-', linewidth=2, markersize=6, color='steelblue')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        logging.error(f"Erreur génération graphique linéaire: {e}")
        return None

def embed_graph_in_markdown(graph_base64: str, caption: str = "Graphique généré") -> str:
    """Emballe un graphique base64 dans du Markdown
    
    Returns:
        Markdown avec image intégrée
    """
    return f"\n\n![{caption}](data:image/png;base64,{graph_base64})\n*{caption}*\n\n"

def process_graph_request(user_message: str) -> Optional[dict]:
    """Traite une demande de graphique et génère l'image
    
    Args:
        user_message: Message de l'utilisateur
    
    Returns:
        Dict avec {
            'graph_base64': str,
            'expression': str,
            'markdown': str
        } ou None
    """
    if not detect_graph_request(user_message):
        return None
    
    # Extraire l'expression
    expression = extract_math_expression(user_message)
    if not expression:
        return None
    
    # Générer le graphique
    graph_base64 = generate_function_graph(
        expression=expression,
        title=f"Graphique de f(x) = {expression}"
    )
    
    if not graph_base64:
        return None
    
    return {
        'graph_base64': graph_base64,
        'expression': expression,
        'markdown': embed_graph_in_markdown(
            graph_base64,
            f"Graphique de f(x) = {expression}"
        )
    }
