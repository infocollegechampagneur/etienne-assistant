import React from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

/**
 * MathRenderer - Composant pour rendre les formules mathématiques LaTeX
 * Supporte le format $...$ pour inline et $$...$$ pour les blocs
 */
const MathRenderer = ({ text }) => {
  if (!text) return null;

  // Regex pour détecter les formules LaTeX
  // $$...$$ pour les blocs (doit être traité en premier)
  // $...$ pour inline (mais pas $$)
  const blockMathRegex = /\$\$([\s\S]*?)\$\$/g;
  const inlineMathRegex = /\$([^$\n]+?)\$/g;

  // Fonction pour traiter le texte et remplacer les formules
  const renderMathText = (inputText) => {
    const parts = [];
    let lastIndex = 0;
    let key = 0;

    // D'abord, traiter les blocs $$...$$
    const textWithBlockMath = inputText.replace(blockMathRegex, (match, formula) => {
      return `%%BLOCK_MATH_${key++}%%${formula}%%END_BLOCK%%`;
    });

    // Puis traiter les formules inline $...$
    const processedText = textWithBlockMath.replace(inlineMathRegex, (match, formula) => {
      return `%%INLINE_MATH_${key++}%%${formula}%%END_INLINE%%`;
    });

    // Maintenant, reconstruire le texte avec les composants React
    const blockRegex = /%%BLOCK_MATH_\d+%%([\s\S]*?)%%END_BLOCK%%/g;
    const inlineRegex = /%%INLINE_MATH_\d+%%(.*?)%%END_INLINE%%/g;

    let tempText = processedText;
    const elements = [];
    let currentKey = 0;

    // Fonction récursive pour traiter le texte
    const processText = (text) => {
      // Chercher le premier placeholder (block ou inline)
      const blockMatch = /%%BLOCK_MATH_\d+%%([\s\S]*?)%%END_BLOCK%%/.exec(text);
      const inlineMatch = /%%INLINE_MATH_\d+%%(.*?)%%END_INLINE%%/.exec(text);

      // Déterminer lequel vient en premier
      let firstMatch = null;
      let isBlock = false;

      if (blockMatch && inlineMatch) {
        if (blockMatch.index < inlineMatch.index) {
          firstMatch = blockMatch;
          isBlock = true;
        } else {
          firstMatch = inlineMatch;
          isBlock = false;
        }
      } else if (blockMatch) {
        firstMatch = blockMatch;
        isBlock = true;
      } else if (inlineMatch) {
        firstMatch = inlineMatch;
        isBlock = false;
      }

      if (!firstMatch) {
        // Plus de formules, retourner le texte restant
        if (text) {
          elements.push(<span key={currentKey++}>{text}</span>);
        }
        return;
      }

      // Ajouter le texte avant la formule
      if (firstMatch.index > 0) {
        elements.push(<span key={currentKey++}>{text.substring(0, firstMatch.index)}</span>);
      }

      // Ajouter la formule
      const formula = firstMatch[1];
      try {
        if (isBlock) {
          elements.push(
            <div key={currentKey++} className="my-2 overflow-x-auto">
              <BlockMath math={formula} />
            </div>
          );
        } else {
          elements.push(
            <InlineMath key={currentKey++} math={formula} />
          );
        }
      } catch (error) {
        // Si erreur de parsing, afficher le texte brut
        console.warn('Erreur LaTeX:', error);
        elements.push(
          <span key={currentKey++} className="text-red-500 font-mono text-sm">
            {isBlock ? `$$${formula}$$` : `$${formula}$`}
          </span>
        );
      }

      // Continuer avec le reste du texte
      processText(text.substring(firstMatch.index + firstMatch[0].length));
    };

    processText(processedText);
    return elements;
  };

  return <span className="math-content">{renderMathText(text)}</span>;
};

/**
 * Fonction utilitaire pour vérifier si un texte contient des formules LaTeX
 */
export const containsLatex = (text) => {
  if (!text) return false;
  return /\$[^$]+\$/.test(text) || /\$\$[\s\S]+?\$\$/.test(text);
};

/**
 * Fonction pour formater un message avec support LaTeX
 * À utiliser avec dangerouslySetInnerHTML si nécessaire
 */
export const formatMessageWithMath = (text) => {
  if (!text) return '';
  
  // Remplacer les formules par des placeholders temporaires pour éviter les conflits
  let processedText = text;
  const mathFormulas = [];
  
  // Extraire les blocs $$...$$
  processedText = processedText.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
    const index = mathFormulas.length;
    mathFormulas.push({ type: 'block', formula });
    return `%%MATH_PLACEHOLDER_${index}%%`;
  });
  
  // Extraire les formules inline $...$
  processedText = processedText.replace(/\$([^$\n]+?)\$/g, (match, formula) => {
    const index = mathFormulas.length;
    mathFormulas.push({ type: 'inline', formula });
    return `%%MATH_PLACEHOLDER_${index}%%`;
  });
  
  return { processedText, mathFormulas };
};

export default MathRenderer;
