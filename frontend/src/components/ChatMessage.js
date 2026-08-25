import React, { useRef, useState, useEffect } from 'react';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Badge } from './ui/badge';
import { formatMessage } from '../utils/formatMessage';
import katex from 'katex';

// Composant pour rendre le contenu avec support LaTeX
const MessageContent = ({ html }) => {
  const containerRef = useRef(null);
  const [renderedHtml, setRenderedHtml] = useState(html);
  
  useEffect(() => {
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    
    const latexElements = tempDiv.querySelectorAll('.latex-inline, .latex-block');
    
    if (latexElements.length > 0) {
      latexElements.forEach((el) => {
        const latexContent = decodeURIComponent(el.getAttribute('data-latex') || '');
        if (!latexContent) return;
        
        const isBlock = el.classList.contains('latex-block');
        
        try {
          const rendered = katex.renderToString(latexContent, {
            throwOnError: false,
            displayMode: isBlock,
            trust: true
          });
          el.innerHTML = rendered;
          el.className = isBlock ? 'katex-block-container' : 'katex-inline-container';
        } catch (error) {
          console.warn('Erreur rendu LaTeX:', error, latexContent);
        }
      });
      
      setRenderedHtml(tempDiv.innerHTML);
    }
  }, [html]);
  
  return (
    <div 
      ref={containerRef}
      className="text-sm leading-relaxed formatted-message" 
      dangerouslySetInnerHTML={{__html: renderedHtml}}
    />
  );
};

// Résumé compact du prompt de correction (masque les instructions techniques)
const parseCorrectionPrompt = (text) => {
  if (!text || !text.includes("TEXTE DE L'ÉLÈVE À CORRIGER")) return null;
  const niveau = text.match(/\*\*Niveau:\*\*\s*([^\n]+)/)?.[1]?.trim();
  const eleve = text.match(/\*\*Élève:\*\*\s*([^\n]+)/)?.[1]?.trim();
  const genre = text.match(/\*\*Genre de texte:\*\*\s*([^\n]+)/)?.[1]?.trim();
  const ps = text.includes('PROFIL SCRIPTEUR (PS) DEMANDÉ') || text.includes('[PROFIL_SCRIPTEUR');
  const plagiat = text.includes('DÉTECTION DE PLAGIAT DEMANDÉE') || text.includes('[DETECTION_PLAGIAT]');
  const consigne = text.match(/\*\*Consigne d'écriture donnée aux élèves[^:]*:\*\*\s*([^\n]+)/)?.[1]?.trim();
  let studentText = text.split("TEXTE DE L'ÉLÈVE À CORRIGER:**")?.[1] || text.split("TEXTE DE L'ÉLÈVE À CORRIGER:")?.[1] || '';
  studentText = studentText.trim();
  return { niveau, eleve, genre, ps, plagiat, consigne, studentText };
};

const CorrectionPromptSummary = ({ info }) => {
  const [showText, setShowText] = useState(false);
  return (
    <div className="text-sm" data-testid="correction-prompt-summary">
      <p className="font-semibold flex items-center gap-1.5">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>
        Correction d'un texte d'élève
      </p>
      <div className="flex flex-wrap gap-1.5 mt-1.5">
        {info.niveau && info.niveau !== 'Non précisé' && (
          <span className="text-[11px] bg-white/20 rounded-full px-2 py-0.5">{info.niveau}</span>
        )}
        {info.eleve && <span className="text-[11px] bg-white/20 rounded-full px-2 py-0.5">👤 {info.eleve}</span>}
        {info.genre && <span className="text-[11px] bg-white/20 rounded-full px-2 py-0.5">{info.genre}</span>}
        {info.ps && <span className="text-[11px] bg-white/20 rounded-full px-2 py-0.5">📋 Profil scripteur</span>}
        {info.plagiat && <span className="text-[11px] bg-white/20 rounded-full px-2 py-0.5">🔍 Détection plagiat</span>}
      </div>
      {info.consigne && (
        <p className="text-xs mt-1.5 opacity-90"><span className="font-medium">Consigne :</span> {info.consigne}</p>
      )}
      {info.studentText && (
        <div className="mt-2">
          <button
            onClick={() => setShowText(!showText)}
            className="text-[11px] underline opacity-90 hover:opacity-100"
            data-testid="toggle-student-text-btn"
          >
            {showText ? 'Masquer le texte de l\'élève' : 'Voir le texte de l\'élève'}
          </button>
          {showText && (
            <p className="text-xs mt-1.5 bg-white/15 rounded-lg p-2 max-h-40 overflow-y-auto whitespace-pre-wrap">{info.studentText}</p>
          )}
        </div>
      )}
    </div>
  );
};

const getTrustBadge = (trustScore) => {
  if (!trustScore) return null;
  const percentage = Math.round(trustScore * 100);
  let variant = 'secondary';
  let text = '';
  
  if (percentage >= 80) {
    variant = 'default';
    text = `Très fiable (${percentage}%)`;
  } else if (percentage >= 60) {
    variant = 'secondary';
    text = `Fiable (${percentage}%)`;
  } else {
    variant = 'destructive';
    text = `Modérément fiable (${percentage}%)`;
  }
  
  return <Badge variant={variant} className="text-xs">{text}</Badge>;
};

export const ChatMessage = ({ msg, prevMsg, downloadDocument, isLoading, currentUser }) => {
  const isCorrection = prevMsg?.isUser && prevMsg?.message?.includes('protocole MEQ');

  // Initiales de l'utilisateur (ex: "Simon Lynch" → "SL")
  const userInitials = currentUser?.full_name
    ? currentUser.full_name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : 'U';

  return (
    <div className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${msg.isUser ? 'order-2' : 'order-1'}`}>
        <div className={`flex items-start gap-3 ${msg.isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          <Avatar className="w-8 h-8">
            <AvatarFallback className={msg.isUser ? 'bg-blue-500 text-white text-xs font-semibold' : 'bg-orange-500 text-white'}>
              {msg.isUser ? userInitials : 'É'}
            </AvatarFallback>
          </Avatar>
          <div className={`rounded-2xl px-4 py-3 ${msg.isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-gray-100 text-gray-900'
          }`}>
            {msg.isUser ? (
              (() => {
                const correctionInfo = parseCorrectionPrompt(msg.message);
                return correctionInfo
                  ? <CorrectionPromptSummary info={correctionInfo} />
                  : <p className="text-sm leading-relaxed">{msg.message}</p>;
              })()
            ) : (
              <MessageContent html={formatMessage(msg.message)} />
            )}
            
            {/* Affichage des images générées */}
            {(msg.images && msg.images.length > 0) ? (
              <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-3">
                {msg.images.map((imgBase64, idx) => (
                  <div key={idx} className="border rounded-lg p-2 bg-white">
                    <img 
                      src={`data:image/png;base64,${imgBase64}`} 
                      alt={`Diagramme ${idx + 1}`} 
                      className="rounded max-w-full h-auto"
                    />
                    <a 
                      href={`data:image/png;base64,${imgBase64}`}
                      download={`angle_${idx + 1}_${Date.now()}.png`}
                      className="inline-block mt-1 text-xs bg-green-100 hover:bg-green-200 text-green-700 px-2 py-1 rounded transition-colors w-full text-center"
                    >
                      Télécharger
                    </a>
                  </div>
                ))}
              </div>
            ) : msg.image_base64 && (
              <div className="mt-3">
                <img 
                  src={`data:image/png;base64,${msg.image_base64}`} 
                  alt="Image générée par IA" 
                  className="rounded-lg max-w-full h-auto"
                />
                <a 
                  href={`data:image/png;base64,${msg.image_base64}`}
                  download={`etienne_image_${Date.now()}.png`}
                  className="inline-block mt-2 text-xs bg-green-100 hover:bg-green-200 text-green-700 px-3 py-1 rounded transition-colors"
                >
                  Télécharger l'image
                </a>
              </div>
            )}
            
            {msg.trust_score && (
              <div className="mt-2">
                {getTrustBadge(msg.trust_score)}
              </div>
            )}

            {!msg.isUser && msg.can_download && (
              <div className="mt-3 pt-2 border-t border-gray-200">
                {isCorrection ? (
                  <>
                    <p className="text-xs text-gray-600 mb-2">Télécharger la correction :</p>
                    <div className="flex flex-col gap-2">
                      <div className="flex gap-1 flex-wrap">
                        <span className="text-[10px] text-gray-500 self-center mr-1">Correction seulement :</span>
                        <button onClick={() => downloadDocument(msg.message, 'Correction Étienne', 'pdf')}
                          className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors">PDF</button>
                        <button onClick={() => downloadDocument(msg.message, 'Correction Étienne', 'docx')}
                          className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded transition-colors">Word</button>
                      </div>
                      <div className="flex gap-1 flex-wrap">
                        <span className="text-[10px] text-gray-500 self-center mr-1">Texte original + correction :</span>
                        <button onClick={() => {
                          const original = prevMsg.message.split("**TEXTE DE L'ÉLÈVE À CORRIGER:**")?.[1] || prevMsg.message.split('**TEXTE:**')?.[1] || '';
                          const combined = `## TEXTE ORIGINAL DE L'ÉLÈVE\n\n${original.trim()}\n\n---\n\n## CORRECTION PAR ÉTIENNE\n\n${msg.message}`;
                          downloadDocument(combined, 'Correction complète', 'pdf');
                        }} className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors">PDF</button>
                        <button onClick={() => {
                          const original = prevMsg.message.split("**TEXTE DE L'ÉLÈVE À CORRIGER:**")?.[1] || prevMsg.message.split('**TEXTE:**')?.[1] || '';
                          const combined = `## TEXTE ORIGINAL DE L'ÉLÈVE\n\n${original.trim()}\n\n---\n\n## CORRECTION PAR ÉTIENNE\n\n${msg.message}`;
                          downloadDocument(combined, 'Correction complète', 'docx');
                        }} className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded transition-colors">Word</button>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-xs text-gray-600 mb-2">Télécharger cette réponse :</p>
                    <div className="flex gap-1 flex-wrap">
                      <button onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'pdf')}
                        className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors" disabled={isLoading}>PDF</button>
                      <button onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'docx')}
                        className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded transition-colors" disabled={isLoading}>Word</button>
                      <button onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'pptx')}
                        className="text-xs bg-orange-100 hover:bg-orange-200 text-orange-700 px-2 py-1 rounded transition-colors" disabled={isLoading}>PowerPoint</button>
                      <button onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'xlsx')}
                        className="text-xs bg-green-100 hover:bg-green-200 text-green-700 px-2 py-1 rounded transition-colors" disabled={isLoading}>Excel</button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
        <div className={`text-xs text-gray-400 mt-1 ${msg.isUser ? 'text-right' : 'text-left'}`}>
          {new Date(msg.timestamp).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
};

export const LoadingIndicator = () => (
  <div className="flex justify-start">
    <div className="flex items-start gap-3">
      <Avatar className="w-8 h-8">
        <AvatarFallback className="bg-orange-500 text-white">É</AvatarFallback>
      </Avatar>
      <div className="bg-gray-100 rounded-2xl px-4 py-3">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
        </div>
      </div>
    </div>
  </div>
);
