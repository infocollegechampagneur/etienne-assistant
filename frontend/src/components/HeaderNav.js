import React from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

export const HeaderNav = ({
  currentUser,
  isAdmin,
  isLicenseAdmin,
  setShowAuthModal,
  setShowAdminPanel,
  setShowLicenseAdminPanel,
  setShowNouveautes,
  setShowProfileModal,
  setShowSiteConfig,
  handleLogout,
  siteConfig
}) => {
  const t = siteConfig?.texts || {};
  const colors = siteConfig?.colors || {};
  const logoImg = siteConfig?.images?.logo_icon;

  const modules = siteConfig?.layout?.modules || {};
  const customLinks = siteConfig?.header?.custom_links || [];

  return (
    <header className="bg-white/90 backdrop-blur-md border-b border-orange-100 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {logoImg ? (
              <img src={logoImg} alt="Logo" className="w-10 h-10 rounded-lg object-cover" />
            ) : (
              <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: `linear-gradient(to right, ${colors.primary || '#f97316'}, ${colors.secondary || '#2563eb'})` }}>
                <span className="text-white font-bold text-lg">{(t.site_title || 'É')[0]}</span>
              </div>
            )}
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{t.site_title || 'Étienne'}</h1>
              <p className="text-sm text-gray-600">{t.site_subtitle || 'Assistant IA pour les membres du personnel scolaire québécois'}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {modules.program_badge !== false && (
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              Programme québécois
            </Badge>
            )}

            {/* Liens personnalisés du header */}
            {customLinks.map((link, i) => (
              link.style === 'solid' ? (
                <Button key={i} size="sm" asChild style={{ background: `linear-gradient(to right, ${colors.primary || '#f97316'}, ${colors.secondary || '#2563eb'})` }}>
                  <a href={link.url} target="_blank" rel="noopener noreferrer">{link.label}</a>
                </Button>
              ) : link.style === 'link' ? (
                <a key={i} href={link.url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium hover:underline" style={{ color: colors.primary || '#f97316' }}>{link.label}</a>
              ) : (
                <Button key={i} size="sm" variant="outline" asChild className="border-gray-300 hover:bg-gray-50">
                  <a href={link.url} target="_blank" rel="noopener noreferrer">{link.label}</a>
                </Button>
              )
            ))}
            
            {currentUser ? (
              <>
                <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                  {currentUser.full_name}
                </Badge>
                {isLicenseAdmin && !isAdmin && (
                  <Button 
                    size="sm" 
                    onClick={() => setShowLicenseAdminPanel(true)}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Ma licence
                  </Button>
                )}
                {isAdmin && (
                  <>
                    <Button 
                      size="sm" 
                      onClick={() => setShowAdminPanel(true)}
                      className="bg-orange-600 hover:bg-orange-700"
                    >
                      Admin
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => setShowSiteConfig(true)}
                      className="border-purple-200 hover:bg-purple-50 text-purple-700"
                      data-testid="site-config-btn"
                    >
                      Design
                    </Button>
                  </>
                )}
                {modules.nouveautes_button !== false && (
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => setShowNouveautes(true)}
                  className="border-orange-200 hover:bg-orange-50"
                  data-testid="nouveautes-btn"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1 text-orange-500"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>
                  Quoi de neuf
                </Button>
                )}
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => setShowProfileModal(true)}
                  className="border-blue-200 hover:bg-blue-50"
                >
                  Profil
                </Button>
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={handleLogout}
                >
                  Déconnexion
                </Button>
              </>
            ) : (
              <Button 
                size="sm"
                onClick={() => setShowAuthModal(true)}
              >
                Connexion
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
