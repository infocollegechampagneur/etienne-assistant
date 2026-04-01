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

export const ChatMessage = ({ msg, prevMsg, downloadDocument, isLoading }) => {
  const isCorrection = prevMsg?.isUser && prevMsg?.message?.includes('protocole MEQ');

  return (
    <div className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${msg.isUser ? 'order-2' : 'order-1'}`}>
        <div className={`flex items-start gap-3 ${msg.isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          <Avatar className="w-8 h-8">
            <AvatarFallback className={msg.isUser ? 'bg-blue-500 text-white' : 'bg-orange-500 text-white'}>
              {msg.isUser ? 'U' : 'É'}
            </AvatarFallback>
          </Avatar>
          <div className={`rounded-2xl px-4 py-3 ${msg.isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-gray-100 text-gray-900'
          }`}>
            {msg.isUser ? (
              <p className="text-sm leading-relaxed">{msg.message}</p>
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
