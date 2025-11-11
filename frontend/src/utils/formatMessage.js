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
 * Nettoie un message pour l'export (enlève les phrases d'intro)
 */
export const cleanMessageForExport = (text) => {
  if (!text) return '';
  
  // Phrases d'intro courantes à enlever
  const introPatterns = [
    /^Bonjour[!,]?\s*/i,
    /^Salut[!,]?\s*/i,
    /^Hello[!,]?\s*/i,
    /^Hi[!,]?\s*/i,
    /^Excellente? (idée|question)[!,]?\s*/i,
    /^Super (question|idée)[!,]?\s*/i,
    /^C'est une (bonne|excellente) question[!,]?\s*/i,
    /^Bonne question[!,]?\s*/i,
    /^Ah[!,]?\s+/i,
    /^Oh[!,]?\s+/i,
    /^D'accord[!,]?\s*/i,
    /^Parfait[!,]?\s*/i,
    /^Très bien[!,]?\s*/i
  ];
  
  // Phrases de fin à enlever
  const outroPatterns = [
    /N'hésite pas .*$/i,
    /N'hésitez pas .*$/i,
    /Si tu as .*$/i,
    /Si vous avez .*$/i,
    /Bonne chance.*$/i,
    /Bon courage.*$/i
  ];
  
  let cleaned = text;
  
  // Enlever les intros (seulement au début)
  for (const pattern of introPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }
  
  // Enlever les phrases de politesse au début des paragraphes
  cleaned = cleaned.replace(/\n\n(Bonjour|Hello|Salut)[!,]?\s*/gi, '\n\n');
  
  // Enlever les outros (seulement à la fin)
  for (const pattern of outroPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }
  
  // Enlever les phrases qui parlent de "on va" au début
  cleaned = cleaned.replace(/^.*on va (se )?dégourdir.*?\.\s*/i, '');
  cleaned = cleaned.replace(/^.*c'est en pratiquant.*?\.\s*/i, '');
  
  return cleaned.trim();
};
