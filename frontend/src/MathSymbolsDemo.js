import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { toast } from 'sonner';

/**
 * Page de démonstration des symboles mathématiques LaTeX
 * Pour aider les enseignants du secondaire québécois
 */
const MathSymbolsDemo = ({ onClose }) => {
  const [copiedSymbol, setCopiedSymbol] = useState(null);

  // Catégories de symboles par niveau scolaire
  const symbolCategories = {
    "Sec 1-2 : Bases": [
      { latex: "\\frac{a}{b}", description: "Fraction", example: "\\frac{3}{4}" },
      { latex: "a^2", description: "Carré", example: "x^2" },
      { latex: "a^3", description: "Cube", example: "x^3" },
      { latex: "\\sqrt{x}", description: "Racine carrée", example: "\\sqrt{16} = 4" },
      { latex: "\\pm", description: "Plus ou moins", example: "\\pm 5" },
      { latex: "\\times", description: "Multiplication", example: "3 \\times 4" },
      { latex: "\\div", description: "Division", example: "12 \\div 3" },
      { latex: "\\neq", description: "Différent de", example: "5 \\neq 3" },
      { latex: "\\approx", description: "Approximativement", example: "\\pi \\approx 3.14" },
      { latex: "\\leq", description: "Inférieur ou égal", example: "x \\leq 5" },
      { latex: "\\geq", description: "Supérieur ou égal", example: "x \\geq 0" },
    ],
    "Sec 2 : Géométrie": [
      { latex: "\\angle", description: "Angle", example: "\\angle ABC = 90°" },
      { latex: "\\triangle", description: "Triangle", example: "\\triangle ABC" },
      { latex: "\\perp", description: "Perpendiculaire", example: "AB \\perp CD" },
      { latex: "\\parallel", description: "Parallèle", example: "AB \\parallel CD" },
      { latex: "\\pi", description: "Pi", example: "C = 2\\pi r" },
      { latex: "a^2 + b^2 = c^2", description: "Pythagore", example: "3^2 + 4^2 = 5^2" },
    ],
    "Sec 3 : Fonctions": [
      { latex: "f(x)", description: "Notation fonction", example: "f(x) = 2x + 3" },
      { latex: "y = mx + b", description: "Fonction affine", example: "y = 2x + 1" },
      { latex: "\\Delta y", description: "Variation de y", example: "\\Delta y = y_2 - y_1" },
      { latex: "x_1, x_2", description: "Indices", example: "x_1 = 3, x_2 = 7" },
      { latex: "\\sin(\\theta)", description: "Sinus", example: "\\sin(30°) = 0.5" },
      { latex: "\\cos(\\theta)", description: "Cosinus", example: "\\cos(60°) = 0.5" },
      { latex: "\\tan(\\theta)", description: "Tangente", example: "\\tan(45°) = 1" },
    ],
    "Sec 4 CST : Quadratiques": [
      { latex: "ax^2 + bx + c", description: "Forme générale", example: "2x^2 + 3x - 5" },
      { latex: "a(x-h)^2 + k", description: "Forme canonique", example: "2(x-1)^2 + 3" },
      { latex: "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}", description: "Formule quadratique", example: "" },
      { latex: "(h, k)", description: "Sommet", example: "Sommet: (2, -3)" },
      { latex: "\\Delta = b^2 - 4ac", description: "Discriminant", example: "\\Delta = 9 - 8 = 1" },
    ],
    "Sec 4-5 TS/SN : Vecteurs": [
      { latex: "\\vec{v}", description: "Vecteur v", example: "\\vec{v} = (3, 4)" },
      { latex: "\\overrightarrow{AB}", description: "Vecteur AB", example: "\\overrightarrow{AB} = B - A" },
      { latex: "\\|\\vec{v}\\|", description: "Norme (longueur)", example: "\\|\\vec{v}\\| = 5" },
      { latex: "\\vec{u} \\cdot \\vec{v}", description: "Produit scalaire", example: "\\vec{u} \\cdot \\vec{v} = 10" },
      { latex: "\\vec{0}", description: "Vecteur nul", example: "\\vec{0} = (0, 0)" },
    ],
    "Sec 5 TS/SN : Avancé": [
      { latex: "a^n", description: "Exposant n", example: "2^5 = 32" },
      { latex: "\\log_a(x)", description: "Logarithme base a", example: "\\log_2(8) = 3" },
      { latex: "\\ln(x)", description: "Logarithme naturel", example: "\\ln(e) = 1" },
      { latex: "e^x", description: "Exponentielle", example: "e^0 = 1" },
      { latex: "\\sum_{i=1}^{n}", description: "Somme", example: "\\sum_{i=1}^{5} i = 15" },
      { latex: "a_n = a_1 + (n-1)d", description: "Suite arithmétique", example: "" },
      { latex: "a_n = a_1 \\cdot r^{n-1}", description: "Suite géométrique", example: "" },
    ],
    "Ensembles de nombres": [
      { latex: "\\mathbb{N}", description: "Naturels", example: "0, 1, 2, 3..." },
      { latex: "\\mathbb{Z}", description: "Entiers", example: "...-2, -1, 0, 1, 2..." },
      { latex: "\\mathbb{Q}", description: "Rationnels", example: "\\frac{1}{2}, \\frac{3}{4}" },
      { latex: "\\mathbb{R}", description: "Réels", example: "\\pi, \\sqrt{2}" },
      { latex: "\\in", description: "Appartient à", example: "3 \\in \\mathbb{N}" },
      { latex: "\\notin", description: "N'appartient pas", example: "-1 \\notin \\mathbb{N}" },
      { latex: "\\subset", description: "Sous-ensemble", example: "\\mathbb{N} \\subset \\mathbb{Z}" },
      { latex: "\\cup", description: "Union", example: "A \\cup B" },
      { latex: "\\cap", description: "Intersection", example: "A \\cap B" },
      { latex: "\\emptyset", description: "Ensemble vide", example: "A \\cap B = \\emptyset" },
    ],
    "Intervalles": [
      { latex: "[a, b]", description: "Fermé", example: "[0, 5]" },
      { latex: "(a, b)", description: "Ouvert", example: "(0, 5)" },
      { latex: "[a, b)", description: "Fermé-ouvert", example: "[0, 5)" },
      { latex: "(a, b]", description: "Ouvert-fermé", example: "(0, 5]" },
      { latex: "[a, +\\infty)", description: "Semi-infini", example: "[0, +\\infty)" },
      { latex: "(-\\infty, b]", description: "Semi-infini", example: "(-\\infty, 5]" },
    ],
  };

  const copyToClipboard = (latex) => {
    // Copier la version pour demander à Étienne
    const textToCopy = `$${latex}$`;
    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopiedSymbol(latex);
      toast.success(`Copié: ${textToCopy}`);
      setTimeout(() => setCopiedSymbol(null), 2000);
    });
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 p-4 flex items-center justify-between z-10">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              📐 Symboles Mathématiques
            </h2>
            <p className="text-sm text-gray-600">
              Cliquez sur un symbole pour le copier et l'utiliser dans vos demandes à Étienne
            </p>
          </div>
          <Button variant="outline" onClick={onClose}>
            ✕ Fermer
          </Button>
        </div>

        {/* Contenu */}
        <div className="p-6 space-y-8">
          {/* Instructions */}
          <Card className="bg-blue-50 border-blue-200">
            <CardContent className="p-4">
              <h3 className="font-semibold text-blue-800 mb-2">💡 Comment utiliser</h3>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Cliquez sur un symbole pour le copier dans le presse-papier</li>
                <li>• Collez-le dans votre message à Étienne avec <kbd className="bg-blue-100 px-1 rounded">Ctrl+V</kbd></li>
                <li>• Étienne comprendra le format LaTeX et l'affichera correctement</li>
                <li>• Exemple: "Explique le théorème $a^2 + b^2 = c^2$"</li>
              </ul>
            </CardContent>
          </Card>

          {/* Catégories de symboles */}
          {Object.entries(symbolCategories).map(([category, symbols]) => (
            <div key={category}>
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
                <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200">
                  {category}
                </Badge>
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {symbols.map((symbol, idx) => (
                  <button
                    key={idx}
                    onClick={() => copyToClipboard(symbol.latex)}
                    className={`
                      p-3 rounded-lg border-2 transition-all text-left
                      ${copiedSymbol === symbol.latex 
                        ? 'border-green-500 bg-green-50' 
                        : 'border-gray-200 hover:border-orange-300 hover:bg-orange-50'
                      }
                    `}
                  >
                    <div className="flex flex-col items-center text-center">
                      {/* Rendu LaTeX */}
                      <div className="text-xl mb-1 min-h-[2rem] flex items-center justify-center">
                        <InlineMath math={symbol.latex} />
                      </div>
                      
                      {/* Description */}
                      <span className="text-xs text-gray-600 font-medium">
                        {symbol.description}
                      </span>
                      
                      {/* Code LaTeX */}
                      <code className="text-[10px] text-gray-400 mt-1 bg-gray-100 px-1 rounded">
                        ${symbol.latex}$
                      </code>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}

          {/* Exemples de formules complètes */}
          <Card className="bg-gradient-to-r from-orange-50 to-blue-50 border-orange-200">
            <CardHeader>
              <CardTitle className="text-lg text-gray-800">
                📝 Exemples de formules complètes
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { name: "Pythagore", formula: "a^2 + b^2 = c^2" },
                  { name: "Aire du cercle", formula: "A = \\pi r^2" },
                  { name: "Équation quadratique", formula: "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}" },
                  { name: "Pente", formula: "m = \\frac{\\Delta y}{\\Delta x} = \\frac{y_2 - y_1}{x_2 - x_1}" },
                  { name: "Somme arithmétique", formula: "S_n = \\frac{n(a_1 + a_n)}{2}" },
                  { name: "Loi des cosinus", formula: "c^2 = a^2 + b^2 - 2ab\\cos(C)" },
                ].map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => copyToClipboard(item.formula)}
                    className="p-4 bg-white rounded-lg border border-gray-200 hover:border-orange-300 hover:shadow-md transition-all text-left"
                  >
                    <div className="text-sm font-medium text-gray-700 mb-2">{item.name}</div>
                    <div className="flex justify-center">
                      <InlineMath math={item.formula} />
                    </div>
                    <div className="text-[10px] text-gray-400 mt-2 text-center">
                      Cliquez pour copier
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Note sur les niveaux */}
          <Card className="bg-yellow-50 border-yellow-200">
            <CardContent className="p-4">
              <h3 className="font-semibold text-yellow-800 mb-2">⚠️ Note importante - Programme MELS</h3>
              <p className="text-sm text-yellow-700">
                Les symboles sont organisés par niveau scolaire selon le Programme de Formation de l'École Québécoise (PFEQ). 
                Étienne respecte strictement le curriculum du secondaire et n'introduira jamais de concepts de niveau CÉGEP ou universitaire.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default MathSymbolsDemo;
