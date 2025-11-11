/**
 * Formatte un message texte pour le rendre plus lisible
 * - Convertit les listes en HTML
 * - Ajoute des espaces entre les paragraphes
 * - Formatte les titres
 */

export const formatMessage = (text) => {
  if (!text) return '';
  
  // Séparer en lignes
  let lines = text.split('\n');
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
