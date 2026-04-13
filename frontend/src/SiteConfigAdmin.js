import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { toast } from 'sonner';

const API = process.env.REACT_APP_BACKEND_URL;

const THEME_PRESETS = {
  default: { label: 'Orange / Bleu (défaut)', primary: '#f97316', secondary: '#2563eb', bg_from: '#fef3e2', bg_via: '#ffffff', bg_to: '#eff6ff', accent: '#ea580c', text_primary: '#1f2937' },
  forest: { label: 'Forêt', primary: '#16a34a', secondary: '#0d9488', bg_from: '#f0fdf4', bg_via: '#ffffff', bg_to: '#f0fdfa', accent: '#15803d', text_primary: '#1f2937' },
  ocean: { label: 'Océan', primary: '#0ea5e9', secondary: '#6366f1', bg_from: '#f0f9ff', bg_via: '#ffffff', bg_to: '#eef2ff', accent: '#0284c7', text_primary: '#1e293b' },
  sunset: { label: 'Coucher de soleil', primary: '#e11d48', secondary: '#f97316', bg_from: '#fff1f2', bg_via: '#ffffff', bg_to: '#fff7ed', accent: '#be123c', text_primary: '#1f2937' },
  royal: { label: 'Royal', primary: '#7c3aed', secondary: '#2563eb', bg_from: '#f5f3ff', bg_via: '#ffffff', bg_to: '#eff6ff', accent: '#6d28d9', text_primary: '#1e1b4b' },
  dark: { label: 'Sombre', primary: '#f97316', secondary: '#3b82f6', bg_from: '#1e293b', bg_via: '#0f172a', bg_to: '#1e293b', accent: '#ea580c', text_primary: '#f1f5f9' },
};

const MODULE_LABELS = {
  hero_section: { label: 'Section Hero (titre + image principale)', icon: '🏠' },
  subjects_sidebar: { label: 'Sidebar Matières scolaires', icon: '📚' },
  features_section: { label: 'Section Fonctionnalités (cartes en bas)', icon: '⭐' },
  quota_widget: { label: 'Widget Requêtes IA (en bas à droite)', icon: '🔋' },
  nouveautes_button: { label: 'Bouton "Quoi de neuf"', icon: '🔔' },
  program_badge: { label: 'Badge "Programme québécois"', icon: '🏷️' },
};

/* ─── Petits composants réutilisables ─── */

const ColorPicker = ({ label, value, onChange }) => (
  <div className="flex items-center gap-3">
    <input type="color" value={value || '#000000'} onChange={e => onChange(e.target.value)} className="w-10 h-10 rounded cursor-pointer border-0 p-0" />
    <div className="flex-1">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      <Input value={value || ''} onChange={e => onChange(e.target.value)} placeholder="#hex" className="mt-1 text-xs font-mono h-8" />
    </div>
  </div>
);

const ImageUploader = ({ label, value, onUpload, onClear }) => {
  const fileRef = useRef(null);
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      {value && (
        <div className="relative inline-block">
          <img src={value} alt={label} className="w-24 h-24 object-cover rounded-lg border" />
          <button onClick={onClear} className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-5 h-5 text-xs flex items-center justify-center hover:bg-red-600">x</button>
        </div>
      )}
      <div><input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onUpload} />
        <Button size="sm" variant="outline" onClick={() => fileRef.current?.click()}>{value ? 'Changer' : 'Uploader'}</Button>
      </div>
    </div>
  );
};

const Slider = ({ label, value, min, max, step, unit, onChange }) => (
  <div className="space-y-1">
    <div className="flex justify-between text-sm">
      <span className="font-medium text-gray-700">{label}</span>
      <span className="text-gray-500">{value}{unit}</span>
    </div>
    <input type="range" min={min} max={max} step={step || 1} value={value} onChange={e => onChange(Number(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
  </div>
);

const Toggle = ({ label, icon, checked, onChange }) => (
  <label className="flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all hover:bg-gray-50" style={{ borderColor: checked ? '#f97316' : '#e5e7eb', background: checked ? '#fff7ed' : '' }}>
    <span className="text-xl">{icon}</span>
    <span className="flex-1 text-sm font-medium text-gray-800">{label}</span>
    <div className={`relative w-11 h-6 rounded-full transition-colors ${checked ? 'bg-orange-500' : 'bg-gray-300'}`} onClick={e => { e.preventDefault(); onChange(!checked); }}>
      <div className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${checked ? 'translate-x-5' : 'translate-x-0.5'}`} />
    </div>
  </label>
);

/* ─── Composant principal ─── */

const SiteConfigAdmin = ({ open, onClose }) => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('layout');
  const token = localStorage.getItem('etienne_token');
  const axiosConfig = { headers: { Authorization: `Bearer ${token}` } };

  useEffect(() => { if (open) loadConfig(); }, [open]);

  const loadConfig = async () => {
    setLoading(true);
    try { const res = await axios.get(`${API}/api/site-config`); setConfig(res.data); }
    catch { toast.error('Erreur chargement configuration'); }
    finally { setLoading(false); }
  };

  const saveConfig = async () => {
    setSaving(true);
    try { await axios.put(`${API}/api/site-config`, config, axiosConfig); toast.success('Sauvegardé ! Rechargez la page pour voir les changements.'); }
    catch { toast.error('Erreur sauvegarde'); }
    finally { setSaving(false); }
  };

  /* helpers */
  const set = (path, value) => {
    setConfig(prev => {
      const copy = JSON.parse(JSON.stringify(prev));
      const keys = path.split('.');
      let obj = copy;
      for (let i = 0; i < keys.length - 1; i++) { if (!obj[keys[i]]) obj[keys[i]] = {}; obj = obj[keys[i]]; }
      obj[keys[keys.length - 1]] = value;
      return copy;
    });
  };

  const updateColor = (key, value) => {
    setConfig(prev => ({ ...prev, colors: { ...prev.colors, [key]: value }, theme_preset: 'custom' }));
  };

  const applyPreset = (key) => {
    const { label, ...colors } = THEME_PRESETS[key];
    setConfig(prev => ({ ...prev, colors: { ...prev.colors, ...colors }, theme_preset: key }));
    toast.success(`Thème "${label}" appliqué`);
  };

  const handleImageUpload = async (e, imageKey) => {
    const file = e.target.files?.[0]; if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    try { const res = await axios.post(`${API}/api/site-config/upload-image`, fd, { headers: { ...axiosConfig.headers, 'Content-Type': 'multipart/form-data' } });
      set(`images.${imageKey}`, res.data.url); toast.success(`Image uploadée`);
    } catch { toast.error('Erreur upload'); }
    e.target.value = '';
  };

  /* header links */
  const addHeaderLink = () => {
    const links = [...(config.header?.custom_links || []), { label: 'Nouveau lien', url: '#', style: 'outline' }];
    set('header.custom_links', links);
  };
  const updateHeaderLink = (idx, field, val) => {
    const links = [...(config.header?.custom_links || [])];
    links[idx] = { ...links[idx], [field]: val };
    set('header.custom_links', links);
  };
  const removeHeaderLink = (idx) => {
    const links = (config.header?.custom_links || []).filter((_, i) => i !== idx);
    set('header.custom_links', links);
  };

  /* footer links */
  const addFooterLink = () => {
    const links = [...(config.footer?.links || []), { label: 'Lien', url: '#' }];
    set('footer.links', links);
  };
  const updateFooterLink = (idx, field, val) => {
    const links = [...(config.footer?.links || [])];
    links[idx] = { ...links[idx], [field]: val };
    set('footer.links', links);
  };
  const removeFooterLink = (idx) => {
    const links = (config.footer?.links || []).filter((_, i) => i !== idx);
    set('footer.links', links);
  };

  if (!open) return null;

  const modules = config?.layout?.modules || {};
  const sizes = config?.layout?.sizes || {};

  return (
    <div className="fixed inset-0 z-[2100] bg-black/60 flex items-center justify-center overflow-y-auto p-4" data-testid="site-config-modal">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[92vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b flex-shrink-0">
          <div>
            <h2 className="text-xl font-bold text-gray-900" data-testid="site-config-title">Éditeur du Site</h2>
            <p className="text-sm text-gray-500">Personnalisez Étienne sans code</p>
          </div>
          <div className="flex gap-2">
            <Button onClick={saveConfig} disabled={saving} className="bg-green-600 hover:bg-green-700 text-white" data-testid="save-config-btn">
              {saving ? 'Sauvegarde...' : 'Sauvegarder'}
            </Button>
            <button onClick={onClose} className="text-2xl text-gray-400 hover:text-red-600 px-2">x</button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {loading || !config ? <p className="text-center py-10 text-gray-500">Chargement...</p> : (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-7 mb-6 text-xs">
                <TabsTrigger value="layout">Modules</TabsTrigger>
                <TabsTrigger value="sizes">Tailles</TabsTrigger>
                <TabsTrigger value="texts">Textes</TabsTrigger>
                <TabsTrigger value="colors">Couleurs</TabsTrigger>
                <TabsTrigger value="images">Images</TabsTrigger>
                <TabsTrigger value="header_footer">Header / Footer</TabsTrigger>
                <TabsTrigger value="advanced">Avancé</TabsTrigger>
              </TabsList>

              {/* ===== MODULES ===== */}
              <TabsContent value="layout">
                <Card>
                  <CardHeader><CardTitle className="text-base">Activer / Désactiver les sections</CardTitle></CardHeader>
                  <CardContent className="space-y-2">
                    {Object.entries(MODULE_LABELS).map(([key, { label, icon }]) => (
                      <Toggle key={key} label={label} icon={icon} checked={modules[key] !== false}
                        onChange={val => set(`layout.modules.${key}`, val)} />
                    ))}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* ===== TAILLES ===== */}
              <TabsContent value="sizes">
                <Card>
                  <CardHeader><CardTitle className="text-base">Dimensions des éléments</CardTitle></CardHeader>
                  <CardContent className="space-y-6">
                    <Slider label="Hauteur de la fenêtre de chat" value={sizes.chat_height || 550} min={300} max={900} step={10} unit="px"
                      onChange={v => set('layout.sizes.chat_height', v)} />
                    <Slider label="Taille de l'image Hero" value={sizes.hero_image_size || 128} min={0} max={256} step={8} unit="px"
                      onChange={v => set('layout.sizes.hero_image_size', v)} />
                    <Slider label="Espacement section Hero" value={sizes.hero_padding || 48} min={0} max={120} step={4} unit="px"
                      onChange={v => set('layout.sizes.hero_padding', v)} />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* ===== TEXTES ===== */}
              <TabsContent value="texts">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">En-tête du site</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      <div><label className="text-sm font-medium text-gray-700">Titre du site</label>
                        <Input value={config.texts?.site_title || ''} onChange={e => set('texts.site_title', e.target.value)} data-testid="input-site-title" /></div>
                      <div><label className="text-sm font-medium text-gray-700">Sous-titre</label>
                        <Input value={config.texts?.site_subtitle || ''} onChange={e => set('texts.site_subtitle', e.target.value)} /></div>
                    </CardContent>
                  </Card>
                  <Card><CardHeader><CardTitle className="text-base">Section Hero</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      <div><label className="text-sm font-medium text-gray-700">Titre (avant le texte en couleur)</label>
                        <Input value={config.texts?.hero_title_before || ''} onChange={e => set('texts.hero_title_before', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Texte en couleur (highlight)</label>
                        <Input value={config.texts?.hero_highlight || ''} onChange={e => set('texts.hero_highlight', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Description</label>
                        <textarea value={config.texts?.hero_description || ''} onChange={e => set('texts.hero_description', e.target.value)} className="w-full border rounded-lg p-2 text-sm h-24 resize-none" /></div>
                    </CardContent>
                  </Card>
                  <Card><CardHeader><CardTitle className="text-base">Page de connexion / Chat</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      <div><label className="text-sm font-medium text-gray-700">Titre connexion requise</label>
                        <Input value={config.texts?.login_title || ''} onChange={e => set('texts.login_title', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Texte connexion requise</label>
                        <Input value={config.texts?.login_text || ''} onChange={e => set('texts.login_text', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Message chat vide</label>
                        <Input value={config.texts?.chat_empty || ''} onChange={e => set('texts.chat_empty', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Sous-texte chat vide</label>
                        <Input value={config.texts?.chat_empty_sub || ''} onChange={e => set('texts.chat_empty_sub', e.target.value)} /></div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* ===== COULEURS ===== */}
              <TabsContent value="colors">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">Thèmes pré-faits</CardTitle></CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {Object.entries(THEME_PRESETS).map(([key, preset]) => (
                          <button key={key} onClick={() => applyPreset(key)}
                            className={`p-3 rounded-xl border-2 text-left transition-all hover:shadow-md ${config.theme_preset === key ? 'border-blue-500 shadow-md' : 'border-gray-200'}`}
                            data-testid={`theme-${key}`}>
                            <div className="flex gap-1 mb-2">
                              <span className="w-6 h-6 rounded-full" style={{ background: preset.primary }}></span>
                              <span className="w-6 h-6 rounded-full" style={{ background: preset.secondary }}></span>
                              <span className="w-6 h-6 rounded-full" style={{ background: preset.bg_from }}></span>
                            </div>
                            <span className="text-sm font-medium">{preset.label}</span>
                          </button>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                  <Card><CardHeader><CardTitle className="text-base">Couleurs personnalisées</CardTitle></CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-4">
                        <ColorPicker label="Couleur primaire" value={config.colors?.primary} onChange={v => updateColor('primary', v)} />
                        <ColorPicker label="Couleur secondaire" value={config.colors?.secondary} onChange={v => updateColor('secondary', v)} />
                        <ColorPicker label="Accent" value={config.colors?.accent} onChange={v => updateColor('accent', v)} />
                        <ColorPicker label="Texte principal" value={config.colors?.text_primary} onChange={v => updateColor('text_primary', v)} />
                        <ColorPicker label="Fond gauche" value={config.colors?.bg_from} onChange={v => updateColor('bg_from', v)} />
                        <ColorPicker label="Fond centre" value={config.colors?.bg_via} onChange={v => updateColor('bg_via', v)} />
                        <ColorPicker label="Fond droite" value={config.colors?.bg_to} onChange={v => updateColor('bg_to', v)} />
                      </div>
                    </CardContent>
                  </Card>
                  {/* Aperçu */}
                  <Card><CardHeader><CardTitle className="text-base">Aperçu</CardTitle></CardHeader>
                    <CardContent>
                      <div className="rounded-xl p-6 border" style={{ background: `linear-gradient(135deg, ${config.colors?.bg_from}, ${config.colors?.bg_via}, ${config.colors?.bg_to})` }}>
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold" style={{ background: config.colors?.primary }}>{(config.texts?.site_title||'É')[0]}</div>
                          <span className="font-bold" style={{ color: config.colors?.text_primary }}>{config.texts?.site_title||'Étienne'}</span>
                        </div>
                        <h3 className="text-lg font-bold mb-2" style={{ color: config.colors?.text_primary }}>
                          {config.texts?.hero_title_before}{' '}<span style={{ color: config.colors?.primary }}>{config.texts?.hero_highlight}</span>
                        </h3>
                        <div className="flex gap-2 mt-3">
                          <button className="px-4 py-2 rounded-lg text-white text-sm" style={{ background: `linear-gradient(135deg, ${config.colors?.primary}, ${config.colors?.secondary})` }}>Bouton primaire</button>
                          <button className="px-4 py-2 rounded-lg text-sm border-2" style={{ borderColor: config.colors?.primary, color: config.colors?.primary }}>Bouton secondaire</button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* ===== IMAGES ===== */}
              <TabsContent value="images">
                <Card><CardHeader><CardTitle className="text-base">Images du site</CardTitle></CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-6">
                      <ImageUploader label="Image Hero (ronde)" value={config.images?.hero_image} onUpload={e => handleImageUpload(e, 'hero_image')} onClear={() => set('images.hero_image', '')} />
                      <ImageUploader label="Logo / Icône" value={config.images?.logo_icon} onUpload={e => handleImageUpload(e, 'logo_icon')} onClear={() => set('images.logo_icon', '')} />
                      <ImageUploader label="Image chat vide" value={config.images?.chat_empty_image} onUpload={e => handleImageUpload(e, 'chat_empty_image')} onClear={() => set('images.chat_empty_image', '')} />
                      <ImageUploader label="Arrière-plan" value={config.images?.background_image} onUpload={e => handleImageUpload(e, 'background_image')} onClear={() => set('images.background_image', '')} />
                    </div>
                    <p className="text-xs text-gray-500 mt-4">Formats : PNG, JPG, WEBP, SVG, GIF. Max 5 MB.</p>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* ===== HEADER / FOOTER ===== */}
              <TabsContent value="header_footer">
                <div className="space-y-5">
                  {/* Header */}
                  <Card>
                    <CardHeader><CardTitle className="text-base">Liens personnalisés dans le header</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      {(config.header?.custom_links || []).map((link, idx) => (
                        <div key={idx} className="flex items-center gap-2 p-2 border rounded-lg bg-gray-50">
                          <Input value={link.label} onChange={e => updateHeaderLink(idx, 'label', e.target.value)} placeholder="Texte" className="flex-1 h-9" />
                          <Input value={link.url} onChange={e => updateHeaderLink(idx, 'url', e.target.value)} placeholder="https://..." className="flex-1 h-9" />
                          <select value={link.style || 'outline'} onChange={e => updateHeaderLink(idx, 'style', e.target.value)} className="border rounded px-2 h-9 text-sm">
                            <option value="outline">Contour</option>
                            <option value="solid">Plein</option>
                            <option value="link">Lien simple</option>
                          </select>
                          <button onClick={() => removeHeaderLink(idx)} className="text-red-500 hover:text-red-700 px-2 text-lg font-bold">x</button>
                        </div>
                      ))}
                      <Button size="sm" variant="outline" onClick={addHeaderLink} data-testid="add-header-link">+ Ajouter un lien</Button>
                    </CardContent>
                  </Card>

                  {/* Footer */}
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base">Footer</CardTitle>
                        <Toggle label="" icon="" checked={config.footer?.enabled || false} onChange={v => set('footer.enabled', v)} />
                      </div>
                    </CardHeader>
                    {config.footer?.enabled && (
                      <CardContent className="space-y-4">
                        <div><label className="text-sm font-medium text-gray-700">Texte principal</label>
                          <Input value={config.footer?.text || ''} onChange={e => set('footer.text', e.target.value)} placeholder="© 2025 Mon Organisation" /></div>
                        <div className="grid grid-cols-2 gap-4">
                          <ColorPicker label="Fond du footer" value={config.footer?.bg_color || '#1f2937'} onChange={v => set('footer.bg_color', v)} />
                          <ColorPicker label="Texte du footer" value={config.footer?.text_color || '#9ca3af'} onChange={v => set('footer.text_color', v)} />
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-700 mb-2 block">Liens du footer</label>
                          {(config.footer?.links || []).map((link, idx) => (
                            <div key={idx} className="flex items-center gap-2 mb-2">
                              <Input value={link.label} onChange={e => updateFooterLink(idx, 'label', e.target.value)} placeholder="Texte" className="flex-1 h-9" />
                              <Input value={link.url} onChange={e => updateFooterLink(idx, 'url', e.target.value)} placeholder="https://..." className="flex-1 h-9" />
                              <button onClick={() => removeFooterLink(idx)} className="text-red-500 hover:text-red-700 px-2 text-lg font-bold">x</button>
                            </div>
                          ))}
                          <Button size="sm" variant="outline" onClick={addFooterLink} data-testid="add-footer-link">+ Ajouter un lien</Button>
                        </div>
                        {/* Aperçu footer */}
                        <div className="rounded-lg p-4 text-center text-sm" style={{ background: config.footer?.bg_color || '#1f2937', color: config.footer?.text_color || '#9ca3af' }}>
                          <p>{config.footer?.text || 'Aperçu du footer'}</p>
                          <div className="flex justify-center gap-4 mt-2">
                            {(config.footer?.links || []).map((l, i) => <span key={i} className="underline">{l.label}</span>)}
                          </div>
                        </div>
                      </CardContent>
                    )}
                  </Card>
                </div>
              </TabsContent>

              {/* ===== AVANCÉ ===== */}
              <TabsContent value="advanced">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">CSS personnalisé</CardTitle></CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-500 mb-3">Ajoutez du CSS pour personnaliser davantage.</p>
                      <textarea value={config.custom_css || ''} onChange={e => set('custom_css', e.target.value)}
                        className="w-full h-48 font-mono text-sm border rounded-lg p-3 bg-gray-50" placeholder={`.App { background: #000; }`} />
                    </CardContent>
                  </Card>
                  <Card><CardHeader><CardTitle className="text-base">Réinitialiser</CardTitle></CardHeader>
                    <CardContent>
                      <Button variant="outline" className="border-red-300 text-red-600 hover:bg-red-50"
                        onClick={() => { loadConfig(); toast.info('Configuration rechargée'); }} data-testid="reset-config-btn">
                        Annuler les modifications
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  );
};

export default SiteConfigAdmin;
