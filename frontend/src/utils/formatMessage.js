/**
 * Formatte un message texte pour le rendre plus lisible
 * - Convertit les listes en HTML
 * - Ajoute des espaces entre les paragraphes
 * - Formatte les titres
 * - Supporte les formules LaTeX (préserve $...$ et $$...$$)
 */

// Fonction pour échapper le HTML tout en préservant le LaTeX
const escapeHtmlExceptLatex = (text) => {
  // Préserver les formules LaTeX ET les spans de correction
  const latexBlocks = [];
  const correctionSpans = [];
  let processedText = text;
  
  // Préserver les spans de correction (couleurs de fautes)
  processedText = processedText.replace(/<span\s+style="[^"]*"[^>]*>.*?<\/span>/g, (match) => {
    const idx = correctionSpans.length;
    correctionSpans.push(match);
    return `%%CORRECTION_SPAN_${idx}%%`;
  });
  
  // Préserver $$...$$ (blocs)
  processedText = processedText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
    const idx = latexBlocks.length;
    latexBlocks.push(match);
    return `%%LATEX_BLOCK_${idx}%%`;
  });
  
  // Préserver $...$ (inline)
  processedText = processedText.replace(/\$([^$\n]+?)\$/g, (match) => {
    const idx = latexBlocks.length;
    latexBlocks.push(match);
    return `%%LATEX_INLINE_${idx}%%`;
  });
  
  // Restaurer les spans de correction
  correctionSpans.forEach((span, idx) => {
    processedText = processedText.replace(`%%CORRECTION_SPAN_${idx}%%`, span);
  });
  
  // Restaurer les formules LaTeX avec des spans spéciaux
  latexBlocks.forEach((formula, idx) => {
    if (formula.startsWith('$$')) {
      const content = formula.slice(2, -2);
      processedText = processedText.replace(
        `%%LATEX_BLOCK_${idx}%%`,
        `<span class="latex-block" data-latex="${encodeURIComponent(content)}">${formula}</span>`
      );
    } else {
      const content = formula.slice(1, -1);
      processedText = processedText.replace(
        `%%LATEX_INLINE_${idx}%%`,
        `<span class="latex-inline" data-latex="${encodeURIComponent(content)}">${formula}</span>`
      );
    }
  });
  
  return processedText;
};

export const formatMessage = (text) => {
  if (!text) return '';
  
  // Nettoyer les images base64 du markdown - elles seront affichées séparément
  let cleanedText = text.replace(/!\[.*?\]\(data:image\/[^)]+\)/g, '');
  
  // Prétraiter le texte pour préserver le LaTeX
  const processedText = escapeHtmlExceptLatex(cleanedText);
  
  // Séparer en lignes
  let lines = processedText.split('\n');
  let formatted = [];
  let inList = false;
  
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    
    // Ligne vide - fermer la liste si nécessaire et ajouter un espace
    if (!trimmed) {
      if (inList) {
        formatted.push('</ul>');
        inList = false;
      }
      formatted.push('<br/>');
      return;
    }
    
    // Titres (lignes commençant par ##, #, ou en gras **)
    if (trimmed.startsWith('##')) {
      if (inList) {
        formatted.push('</ul>');
        inList = false;
      }
      const title = trimmed.replace(/^##\s*/, '').replace(/\*\*/g, '');
      formatted.push(`<h3 class="message-title">${title}</h3>`);
      return;
    }
    
    if (trimmed.startsWith('#')) {
      if (inList) {
        formatted.push('</ul>');
        inList = false;
      }
      const title = trimmed.replace(/^#\s*/, '').replace(/\*\*/g, '');
      formatted.push(`<h2 class="message-heading">${title}</h2>`);
      return;
    }
    
    // Lignes de liste (commençant par -, *, •, ou numéro)
    if (trimmed.match(/^[-*•]\s/) || trimmed.match(/^\d+[\.)]\s/)) {
      if (!inList) {
        formatted.push('<ul class="message-list">');
        inList = true;
      }
      const content = trimmed.replace(/^[-*•]\s/, '').replace(/^\d+[\.)]\s/, '');
      formatted.push(`<li>${content}</li>`);
      return;
    }
    
    // Texte en gras **texte**
    let processedLine = trimmed;
    processedLine = processedLine.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Paragraphe normal
    if (inList && !trimmed.match(/^[-*•]\s/)) {
      formatted.push('</ul>');
      inList = false;
    }
    
    formatted.push(`<p class="message-paragraph">${processedLine}</p>`);
  });
  
  // Fermer la liste si encore ouverte
  if (inList) {
    formatted.push('</ul>');
  }
  
  return formatted.join('');
};

/**
 * Nettoie un message pour l'export (enlève les phrases d'intro et de présentation)
 */
export const cleanMessageForExport = (text) => {
  if (!text) return '';
  
  let cleaned = text;
  
  // Enlever TOUTES les phrases (jusqu'au point) qui contiennent ces mots-clés
  const unwantedPhrases = [
    "C'est Étienne", "C'est Etienne", "votre assistant", "assistant pédagogique", 
    "assistant pedagogique", "Je suis ravi", "Je suis là pour", "ravi de vous aider",
    "I am Etienne", "I'm Etienne", "your educational assistant", "I'm here to help",
    "I'm happy to help", "I'm glad to help", "happy to help you", "glad to help you"
  ];
  
  // Supprimer toutes les phrases contenant ces mots-clés
  for (const phrase of unwantedPhrases) {
    // Regex pour supprimer la phrase entière contenant le mot-clé (jusqu'au point)
    const regex = new RegExp(`[^.]*${phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[^.]*\\.\\s*`, 'gi');
    cleaned = cleaned.replace(regex, '');
  }
  
  // Phrases d'intro courantes à enlever (début du texte)
  const introPatterns = [
    /^Bonjour[!,]?\s*/i,
    /^Salut[!,]?\s*/i,
    /^Hello[!,]?\s*/i,
    /^Hi[!,]?\s*/i,
    /^Hey[!,]?\s*/i,
    /^Excellente? (idée|question)[!,]?\s*/i,
    /^Super (question|idée|idee)[!,]?\s*/i,
    /^C'est une (bonne|excellente) (question|idée|idee)[!,]?\s*/i,
    /^Bonne question[!,]?\s*/i,
    /^Great question[!,]?\s*/i,
    /^Good question[!,]?\s*/i,
    /^Absolutely[!,]?\s*/i,
    /^Of course[!,]?\s*/i,
    /^Bien sûr[!,]?\s*/i,
    /^Ah[!,]?\s+/i,
    /^Oh[!,]?\s+/i,
    /^D'accord[!,]?\s*/i,
    /^Parfait[!,]?\s*/i,
    /^Très bien[!,]?\s*/i,
    /^Perfect[!,]?\s*/i
  ];
  
  for (const pattern of introPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }
  
  // Enlever les phrases d'encouragement au début
  const encouragementPatterns = [
    /^[^.\n]*?on va (se )?dégourdir[^.\n]*?\.?\s*/i,
    /^[^.\n]*?c'est en pratiquant[^.\n]*?\.?\s*/i,
    /^[^.\n]*?let's dive in[^.\n]*?\.?\s*/i,
    /^[^.\n]*?let's get started[^.\n]*?\.?\s*/i,
    /^[^.\n]*?commençons[^.\n]*?\.?\s*/i
  ];
  
  for (const pattern of encouragementPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }
  
  // Enlever les phrases de politesse au début des paragraphes
  cleaned = cleaned.replace(/\n\n(Bonjour|Hello|Salut|Hi)[!,]?\s*/gi, '\n\n');
  
  // Phrases de fin à enlever (fin du texte)
  const outroPatterns = [
    /N'hésite pas si tu as.*$/is,
    /N'hésitez pas si vous avez.*$/is,
    /N'hésite pas à.*$/is,
    /N'hésitez pas à.*$/is,
    /Si tu as (d'autres )?questions?.*$/is,
    /Si vous avez (d'autres )?questions?.*$/is,
    /Bonne chance.*$/is,
    /Bon courage.*$/is,
    /Good luck.*$/is,
    /Feel free to ask.*$/is,
    /Don't hesitate.*$/is,
    /If you have any questions.*$/is,
    /Gêne-toi pas.*$/is,
    /Gene-toi pas.*$/is,
    /As-tu d'autres questions.*$/is
  ];
  
  for (const pattern of outroPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }
  
  // Enlever les lignes vides multiples
  cleaned = cleaned.replace(/\n\n\n+/g, '\n\n');
  
  return cleaned.trim();
};


// Utilitaire pour nettoyer le LaTeX pour impression
export const cleanLatexForPrint = (text) => {
  if (!text) return '';
  let cleaned = text;
  
  // Fractions imbriquées
  for (let i = 0; i < 5; i++) {
    cleaned = cleaned.replace(/\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, '($1/$2)');
  }
  
  // Racines
  cleaned = cleaned.replace(/\\sqrt\[(\d+)\]\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, '∜($2)');
  cleaned = cleaned.replace(/\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, '√($1)');
  
  // Délimiteurs LaTeX
  cleaned = cleaned.replace(/\$\$/g, '');
  cleaned = cleaned.replace(/\$/g, '');
  cleaned = cleaned.replace(/\\\[/g, '');
  cleaned = cleaned.replace(/\\\]/g, '');
  cleaned = cleaned.replace(/\\\(/g, '');
  cleaned = cleaned.replace(/\\\)/g, '');
  
  // Puissances
  const superscripts = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','n':'ⁿ'};
  Object.entries(superscripts).forEach(([k, v]) => {
    cleaned = cleaned.replace(new RegExp(`\\^\\{${k}\\}`, 'g'), v);
    if (k !== 'n') {
      cleaned = cleaned.replace(new RegExp(`\\^${k}(?![0-9])`, 'g'), v);
    } else {
      cleaned = cleaned.replace(new RegExp(`\\^${k}(?![a-z])`, 'g'), v);
    }
  });
  cleaned = cleaned.replace(/\^{([^}]*)}/g, '^($1)');
  
  // Indices
  const subscripts = {'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉'};
  Object.entries(subscripts).forEach(([k, v]) => {
    cleaned = cleaned.replace(new RegExp(`_\\{${k}\\}`, 'g'), v);
    cleaned = cleaned.replace(new RegExp(`_${k}(?![0-9])`, 'g'), v);
  });
  cleaned = cleaned.replace(/_{([^}]*)}/g, '_($1)');
  
  // Symboles mathématiques
  const symbols = {
    '\\\\times':'×', '\\\\div':'÷', '\\\\pm':'±', '\\\\mp':'∓',
    '\\\\le(?:q)?':'≤', '\\\\ge(?:q)?':'≥', '\\\\neq':'≠', '\\\\ne(?![a-z])':'≠',
    '\\\\approx':'≈', '\\\\sim':'~', '\\\\infty':'∞', '\\\\pi':'π',
    '\\\\alpha':'α', '\\\\beta':'β', '\\\\gamma':'γ', '\\\\delta':'δ',
    '\\\\epsilon':'ε', '\\\\theta':'θ', '\\\\lambda':'λ', '\\\\mu':'μ',
    '\\\\sigma':'σ', '\\\\omega':'ω', '\\\\Delta':'Δ', '\\\\Sigma':'Σ',
    '\\\\Omega':'Ω', '\\\\sum':'Σ', '\\\\prod':'Π', '\\\\int':'∫',
    '\\\\rightarrow':'→', '\\\\leftarrow':'←', '\\\\Rightarrow':'⇒',
    '\\\\Leftarrow':'⇐', '\\\\leftrightarrow':'↔', '\\\\therefore':'∴',
    '\\\\because':'∵', '\\\\forall':'∀', '\\\\exists':'∃', '\\\\in':'∈',
    '\\\\notin':'∉', '\\\\subset':'⊂', '\\\\supset':'⊃', '\\\\cup':'∪',
    '\\\\cap':'∩', '\\\\emptyset':'∅', '\\\\angle':'∠', '\\\\perp':'⊥',
    '\\\\parallel':'∥', '\\\\degree':'°', '\\\\circ':'°', '\\\\cdot':'·',
    '\\\\ldots':'...', '\\\\dots':'...', '\\\\vdots':'⋮', '\\\\ddots':'⋱'
  };
  Object.entries(symbols).forEach(([pattern, replacement]) => {
    cleaned = cleaned.replace(new RegExp(pattern, 'g'), replacement);
  });
  
  // Espaces LaTeX
  cleaned = cleaned.replace(/\\,/g, ' ');
  cleaned = cleaned.replace(/\\;/g, ' ');
  cleaned = cleaned.replace(/\\:/g, ' ');
  cleaned = cleaned.replace(/\\!/g, '');
  cleaned = cleaned.replace(/\\ /g, ' ');
  cleaned = cleaned.replace(/\\quad/g, '  ');
  cleaned = cleaned.replace(/\\qquad/g, '    ');
  cleaned = cleaned.replace(/~/g, ' ');
  
  // Commandes de texte
  cleaned = cleaned.replace(/\\text\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\textbf\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\textit\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\mathrm\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\mathbf\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\mathit\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\mathsf\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\overline\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\underline\{([^}]*)\}/g, '$1');
  cleaned = cleaned.replace(/\\left/g, '');
  cleaned = cleaned.replace(/\\right/g, '');
  cleaned = cleaned.replace(/\\big/g, '');
  cleaned = cleaned.replace(/\\Big/g, '');
  cleaned = cleaned.replace(/\\begin\{[^}]*\}/g, '');
  cleaned = cleaned.replace(/\\end\{[^}]*\}/g, '');
  cleaned = cleaned.replace(/\\hline/g, '');
  cleaned = cleaned.replace(/&/g, ' | ');
  cleaned = cleaned.replace(/\\\\/g, ' ');
  
  // Commandes inconnues
  cleaned = cleaned.replace(/\\([a-zA-Z]+)/g, '$1');
  
  // Accolades
  cleaned = cleaned.replace(/\{\}/g, '');
  for (let i = 0; i < 3; i++) {
    cleaned = cleaned.replace(/\{([^{}]*)\}/g, '$1');
  }
  
  // Nettoyage final
  cleaned = cleaned.replace(/\(\)/g, '');
  cleaned = cleaned.replace(/\(\s*\)/g, '');
  cleaned = cleaned.replace(/\s+/g, ' ');
  
  return cleaned.trim();
};
