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

/* ─── Petits composants ─── */
const ColorPicker = ({ label, value, onChange }) => (
  <div className="flex items-center gap-3">
    <input type="color" value={value || '#000000'} onChange={e => onChange(e.target.value)} className="w-10 h-10 rounded cursor-pointer border-0 p-0" />
    <div className="flex-1"><label className="text-sm font-medium text-gray-700">{label}</label>
      <Input value={value || ''} onChange={e => onChange(e.target.value)} placeholder="#hex" className="mt-1 text-xs font-mono h-8" /></div>
  </div>
);
const ImageUploader = ({ label, value, onUpload, onClear }) => {
  const r = useRef(null);
  return (<div className="space-y-2"><label className="text-sm font-medium text-gray-700">{label}</label>
    {value && <div className="relative inline-block"><img src={value} alt={label} className="w-24 h-24 object-cover rounded-lg border" />
      <button onClick={onClear} className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-5 h-5 text-xs flex items-center justify-center hover:bg-red-600">x</button></div>}
    <div><input ref={r} type="file" accept="image/*" className="hidden" onChange={onUpload} />
      <Button size="sm" variant="outline" onClick={() => r.current?.click()}>{value ? 'Changer' : 'Uploader'}</Button></div></div>);
};
const Slider = ({ label, value, min, max, step, unit, onChange }) => (
  <div className="space-y-1"><div className="flex justify-between text-sm"><span className="font-medium text-gray-700">{label}</span><span className="text-gray-500">{value}{unit}</span></div>
    <input type="range" min={min} max={max} step={step || 1} value={value} onChange={e => onChange(Number(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" /></div>
);
const Toggle = ({ label, icon, checked, onChange, small }) => (
  <label className={`flex items-center gap-3 ${small ? 'p-2' : 'p-3'} rounded-lg border cursor-pointer transition-all hover:bg-gray-50`} style={{ borderColor: checked ? '#f97316' : '#e5e7eb', background: checked ? '#fff7ed' : '' }}>
    {icon && <span className="text-lg">{icon}</span>}
    <span className={`flex-1 ${small ? 'text-xs' : 'text-sm'} font-medium text-gray-800`}>{label}</span>
    <div className={`relative ${small ? 'w-9 h-5' : 'w-11 h-6'} rounded-full transition-colors ${checked ? 'bg-orange-500' : 'bg-gray-300'}`} onClick={e => { e.preventDefault(); onChange(!checked); }}>
      <div className={`absolute top-0.5 ${small ? 'w-4 h-4' : 'w-5 h-5'} bg-white rounded-full shadow transition-transform ${checked ? (small ? 'translate-x-4' : 'translate-x-5') : 'translate-x-0.5'}`} /></div>
  </label>
);

/* ─── Composant principal ─── */
const SiteConfigAdmin = ({ open, onClose }) => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('layout');
  const token = localStorage.getItem('etienne_token');
  const ax = { headers: { Authorization: `Bearer ${token}` } };

  useEffect(() => { if (open) loadConfig(); }, [open]);
  const loadConfig = async () => { setLoading(true); try { const r = await axios.get(`${API}/api/site-config`); setConfig(r.data); } catch { toast.error('Erreur'); } finally { setLoading(false); } };
  const saveConfig = async () => { setSaving(true); try { await axios.put(`${API}/api/site-config`, config, ax); toast.success('Sauvegardé ! Rechargez pour voir.'); } catch { toast.error('Erreur'); } finally { setSaving(false); } };

  const set = (path, value) => {
    setConfig(prev => { const c = JSON.parse(JSON.stringify(prev)); const k = path.split('.'); let o = c; for (let i = 0; i < k.length - 1; i++) { if (!o[k[i]]) o[k[i]] = {}; o = o[k[i]]; } o[k[k.length - 1]] = value; return c; });
  };
  const updateColor = (key, val) => setConfig(p => ({ ...p, colors: { ...p.colors, [key]: val }, theme_preset: 'custom' }));
  const applyPreset = (k) => { const { label, ...c } = THEME_PRESETS[k]; setConfig(p => ({ ...p, colors: { ...p.colors, ...c }, theme_preset: k })); toast.success(`Thème "${label}"`); };
  const handleImageUpload = async (e, key) => { const f = e.target.files?.[0]; if (!f) return; const fd = new FormData(); fd.append('file', f);
    try { const r = await axios.post(`${API}/api/site-config/upload-image`, fd, { headers: { ...ax.headers, 'Content-Type': 'multipart/form-data' } }); set(`images.${key}`, r.data.url); toast.success('Uploadée'); } catch { toast.error('Erreur'); } e.target.value = ''; };

  const addHeaderLink = () => set('header.custom_links', [...(config.header?.custom_links || []), { label: 'Lien', url: '#', style: 'outline' }]);
  const updateHeaderLink = (i, f, v) => { const l = [...(config.header?.custom_links || [])]; l[i] = { ...l[i], [f]: v }; set('header.custom_links', l); };
  const removeHeaderLink = (i) => set('header.custom_links', (config.header?.custom_links || []).filter((_, j) => j !== i));
  const addFooterLink = () => set('footer.links', [...(config.footer?.links || []), { label: 'Lien', url: '#' }]);
  const updateFooterLink = (i, f, v) => { const l = [...(config.footer?.links || [])]; l[i] = { ...l[i], [f]: v }; set('footer.links', l); };
  const removeFooterLink = (i) => set('footer.links', (config.footer?.links || []).filter((_, j) => j !== i));

  if (!open) return null;
  const modules = config?.layout?.modules || {};
  const sizes = config?.layout?.sizes || {};
  const chatTabs = config?.layout?.chat_tabs || {};
  const TAB_ORDER = ['plans_cours', 'evaluations', 'activites', 'ressources', 'outils'];
  const sortedTabs = TAB_ORDER.filter(k => chatTabs[k]).sort((a, b) => (chatTabs[a]?.order ?? 0) - (chatTabs[b]?.order ?? 0));

  return (
    <div className="fixed inset-0 z-[2100] bg-black/60 flex items-center justify-center overflow-y-auto p-4" data-testid="site-config-modal">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[92vh] flex flex-col">
        <div className="flex items-center justify-between p-5 border-b flex-shrink-0">
          <div><h2 className="text-xl font-bold text-gray-900" data-testid="site-config-title">Éditeur du Site</h2>
            <p className="text-sm text-gray-500">Personnalisez chaque élément sans code</p></div>
          <div className="flex gap-2">
            <Button onClick={saveConfig} disabled={saving} className="bg-green-600 hover:bg-green-700 text-white" data-testid="save-config-btn">{saving ? 'Sauvegarde...' : 'Sauvegarder'}</Button>
            <button onClick={onClose} className="text-2xl text-gray-400 hover:text-red-600 px-2">x</button></div>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {loading || !config ? <p className="text-center py-10 text-gray-500">Chargement...</p> : (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="flex flex-wrap gap-1 mb-6">
                {[['layout','Sections'],['chat','Chat'],['sizes','Tailles'],['texts','Textes'],['colors','Couleurs'],['images','Images'],['header_footer','Header/Footer'],['advanced','Avancé']].map(([v,l]) => (
                  <TabsTrigger key={v} value={v} className="text-xs px-3">{l}</TabsTrigger>
                ))}
              </TabsList>

              {/* ===== SECTIONS / MODULES ===== */}
              <TabsContent value="layout">
                <div className="space-y-4">
                  <Card><CardHeader><CardTitle className="text-base">Sections principales</CardTitle></CardHeader>
                    <CardContent className="space-y-2">
                      {[['hero_section','Section Hero (titre + image)','🏠'],['subjects_sidebar','Sidebar Matières scolaires','📚'],['features_section','Cartes Fonctionnalités (bas de page)','⭐'],['quota_widget','Widget Requêtes IA','🔋']].map(([k,l,i]) => (
                        <Toggle key={k} label={l} icon={i} checked={modules[k] !== false} onChange={v => set(`layout.modules.${k}`, v)} />
                      ))}
                    </CardContent>
                  </Card>
                  <Card><CardHeader><CardTitle className="text-base">Boutons du header</CardTitle></CardHeader>
                    <CardContent className="space-y-2">
                      {[['program_badge','Badge "Programme québécois"','🏷️'],['nouveautes_button','Bouton "Quoi de neuf"','🔔']].map(([k,l,i]) => (
                        <Toggle key={k} label={l} icon={i} checked={modules[k] !== false} onChange={v => set(`layout.modules.${k}`, v)} />
                      ))}
                      <Toggle label='Badge nom utilisateur' icon='👤' checked={config.header?.show_user_badge !== false} onChange={v => set('header.show_user_badge', v)} />
                      <Toggle label='Bouton "Profil"' icon='⚙️' checked={config.header?.show_profile_button !== false} onChange={v => set('header.show_profile_button', v)} />
                      <Toggle label='Bouton "Déconnexion"' icon='🚪' checked={config.header?.show_logout_button !== false} onChange={v => set('header.show_logout_button', v)} />
                    </CardContent>
                  </Card>
                  <Card><CardHeader><CardTitle className="text-base">Boutons de la zone de chat</CardTitle></CardHeader>
                    <CardContent className="space-y-2">
                      {[['correction_button','Bouton "Corriger un texte"','📝'],['print_button','Bouton "Imprimer"','🖨️'],['fullscreen_button','Bouton "Plein écran"','🔲'],['file_upload_info','Texte formats de fichiers','📎']].map(([k,l,i]) => (
                        <Toggle key={k} label={l} icon={i} checked={modules[k] !== false} onChange={v => set(`layout.modules.${k}`, v)} />
                      ))}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* ===== ONGLETS DU CHAT ===== */}
              <TabsContent value="chat">
                <div className="space-y-4">
                  <p className="text-sm text-gray-500">Personnalisez chaque onglet du chat : renommez, masquez, changez la description et le texte d'aide.</p>
                  {sortedTabs.map((tabKey) => {
                    const tab = chatTabs[tabKey] || {};
                    return (
                      <Card key={tabKey} className={!tab.visible && tab.visible !== undefined ? 'opacity-60' : ''}>
                        <CardContent className="p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <h4 className="font-semibold text-sm text-gray-900">{tab.title || tabKey}</h4>
                            <div className="flex items-center gap-3">
                              <span className="text-xs text-gray-400">Ordre: </span>
                              <select value={tab.order ?? 0} onChange={e => set(`layout.chat_tabs.${tabKey}.order`, Number(e.target.value))} className="border rounded px-2 py-1 text-xs w-16">
                                {[0,1,2,3,4].map(n => <option key={n} value={n}>{n + 1}</option>)}
                              </select>
                              <Toggle label="" small checked={tab.visible !== false} onChange={v => set(`layout.chat_tabs.${tabKey}.visible`, v)} />
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div><label className="text-xs text-gray-600">Titre de l'onglet</label>
                              <Input value={tab.title || ''} onChange={e => set(`layout.chat_tabs.${tabKey}.title`, e.target.value)} className="h-8 text-sm" /></div>
                            <div><label className="text-xs text-gray-600">Icône (emoji)</label>
                              <Input value={tab.icon || ''} onChange={e => set(`layout.chat_tabs.${tabKey}.icon`, e.target.value)} className="h-8 text-sm" placeholder="📚" /></div>
                          </div>
                          <div><label className="text-xs text-gray-600">Description (sous le titre)</label>
                            <Input value={tab.description || ''} onChange={e => set(`layout.chat_tabs.${tabKey}.description`, e.target.value)} className="h-8 text-sm" /></div>
                          <div><label className="text-xs text-gray-600">Placeholder de la zone de texte</label>
                            <Input value={tab.placeholder || ''} onChange={e => set(`layout.chat_tabs.${tabKey}.placeholder`, e.target.value)} className="h-8 text-sm" /></div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </TabsContent>

              {/* ===== TAILLES ===== */}
              <TabsContent value="sizes">
                <Card><CardHeader><CardTitle className="text-base">Dimensions</CardTitle></CardHeader>
                  <CardContent className="space-y-6">
                    <Slider label="Hauteur fenêtre de chat" value={sizes.chat_height || 550} min={300} max={900} step={10} unit="px" onChange={v => set('layout.sizes.chat_height', v)} />
                    <Slider label="Taille image Hero" value={sizes.hero_image_size || 128} min={0} max={300} step={4} unit="px" onChange={v => set('layout.sizes.hero_image_size', v)} />
                    <Slider label="Espacement section Hero" value={sizes.hero_padding || 48} min={0} max={120} step={4} unit="px" onChange={v => set('layout.sizes.hero_padding', v)} />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* ===== TEXTES ===== */}
              <TabsContent value="texts">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">En-tête</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      <div><label className="text-sm font-medium text-gray-700">Titre du site</label><Input value={config.texts?.site_title || ''} onChange={e => set('texts.site_title', e.target.value)} data-testid="input-site-title" /></div>
                      <div><label className="text-sm font-medium text-gray-700">Sous-titre</label><Input value={config.texts?.site_subtitle || ''} onChange={e => set('texts.site_subtitle', e.target.value)} /></div>
                    </CardContent></Card>
                  <Card><CardHeader><CardTitle className="text-base">Section Hero</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      <div><label className="text-sm font-medium text-gray-700">Titre (avant couleur)</label><Input value={config.texts?.hero_title_before || ''} onChange={e => set('texts.hero_title_before', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Texte en couleur</label><Input value={config.texts?.hero_highlight || ''} onChange={e => set('texts.hero_highlight', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Description</label><textarea value={config.texts?.hero_description || ''} onChange={e => set('texts.hero_description', e.target.value)} className="w-full border rounded-lg p-2 text-sm h-24 resize-none" /></div>
                    </CardContent></Card>
                  <Card><CardHeader><CardTitle className="text-base">Page connexion / Chat</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      <div><label className="text-sm font-medium text-gray-700">Titre connexion</label><Input value={config.texts?.login_title || ''} onChange={e => set('texts.login_title', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Texte connexion</label><Input value={config.texts?.login_text || ''} onChange={e => set('texts.login_text', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Message chat vide</label><Input value={config.texts?.chat_empty || ''} onChange={e => set('texts.chat_empty', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Sous-texte chat vide</label><Input value={config.texts?.chat_empty_sub || ''} onChange={e => set('texts.chat_empty_sub', e.target.value)} /></div>
                      <div><label className="text-sm font-medium text-gray-700">Texte formats fichiers</label><Input value={config.texts?.file_formats || ''} onChange={e => set('texts.file_formats', e.target.value)} placeholder="📎 Formats: PDF, Word, Excel..." /></div>
                    </CardContent></Card>
                </div>
              </TabsContent>

              {/* ===== COULEURS ===== */}
              <TabsContent value="colors">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">Thèmes</CardTitle></CardHeader>
                    <CardContent><div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(THEME_PRESETS).map(([k, p]) => (
                        <button key={k} onClick={() => applyPreset(k)} className={`p-3 rounded-xl border-2 text-left transition-all hover:shadow-md ${config.theme_preset === k ? 'border-blue-500 shadow-md' : 'border-gray-200'}`} data-testid={`theme-${k}`}>
                          <div className="flex gap-1 mb-2"><span className="w-6 h-6 rounded-full" style={{ background: p.primary }}></span><span className="w-6 h-6 rounded-full" style={{ background: p.secondary }}></span><span className="w-6 h-6 rounded-full" style={{ background: p.bg_from }}></span></div>
                          <span className="text-sm font-medium">{p.label}</span></button>))}
                    </div></CardContent></Card>
                  <Card><CardHeader><CardTitle className="text-base">Personnalisées</CardTitle></CardHeader>
                    <CardContent><div className="grid grid-cols-2 gap-4">
                      <ColorPicker label="Primaire" value={config.colors?.primary} onChange={v => updateColor('primary', v)} />
                      <ColorPicker label="Secondaire" value={config.colors?.secondary} onChange={v => updateColor('secondary', v)} />
                      <ColorPicker label="Accent" value={config.colors?.accent} onChange={v => updateColor('accent', v)} />
                      <ColorPicker label="Texte" value={config.colors?.text_primary} onChange={v => updateColor('text_primary', v)} />
                      <ColorPicker label="Fond gauche" value={config.colors?.bg_from} onChange={v => updateColor('bg_from', v)} />
                      <ColorPicker label="Fond centre" value={config.colors?.bg_via} onChange={v => updateColor('bg_via', v)} />
                      <ColorPicker label="Fond droite" value={config.colors?.bg_to} onChange={v => updateColor('bg_to', v)} />
                    </div></CardContent></Card>
                  <Card><CardHeader><CardTitle className="text-base">Aperçu</CardTitle></CardHeader>
                    <CardContent><div className="rounded-xl p-6 border" style={{ background: `linear-gradient(135deg, ${config.colors?.bg_from}, ${config.colors?.bg_via}, ${config.colors?.bg_to})` }}>
                      <div className="flex items-center gap-2 mb-3"><div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold" style={{ background: config.colors?.primary }}>{(config.texts?.site_title||'É')[0]}</div>
                        <span className="font-bold" style={{ color: config.colors?.text_primary }}>{config.texts?.site_title||'Étienne'}</span></div>
                      <h3 className="text-lg font-bold mb-2" style={{ color: config.colors?.text_primary }}>{config.texts?.hero_title_before}{' '}<span style={{ color: config.colors?.primary }}>{config.texts?.hero_highlight}</span></h3>
                      <div className="flex gap-2 mt-3"><button className="px-4 py-2 rounded-lg text-white text-sm" style={{ background: `linear-gradient(135deg, ${config.colors?.primary}, ${config.colors?.secondary})` }}>Primaire</button>
                        <button className="px-4 py-2 rounded-lg text-sm border-2" style={{ borderColor: config.colors?.primary, color: config.colors?.primary }}>Secondaire</button></div>
                    </div></CardContent></Card>
                </div>
              </TabsContent>

              {/* ===== IMAGES ===== */}
              <TabsContent value="images">
                <Card><CardHeader><CardTitle className="text-base">Images</CardTitle></CardHeader>
                  <CardContent><div className="grid grid-cols-2 gap-6">
                    <ImageUploader label="Image Hero" value={config.images?.hero_image} onUpload={e => handleImageUpload(e, 'hero_image')} onClear={() => set('images.hero_image', '')} />
                    <ImageUploader label="Logo" value={config.images?.logo_icon} onUpload={e => handleImageUpload(e, 'logo_icon')} onClear={() => set('images.logo_icon', '')} />
                    <ImageUploader label="Chat vide" value={config.images?.chat_empty_image} onUpload={e => handleImageUpload(e, 'chat_empty_image')} onClear={() => set('images.chat_empty_image', '')} />
                    <ImageUploader label="Arrière-plan" value={config.images?.background_image} onUpload={e => handleImageUpload(e, 'background_image')} onClear={() => set('images.background_image', '')} />
                  </div><p className="text-xs text-gray-500 mt-4">PNG, JPG, WEBP, SVG, GIF. Max 5 MB.</p></CardContent></Card>
              </TabsContent>

              {/* ===== HEADER / FOOTER ===== */}
              <TabsContent value="header_footer">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">Liens du header</CardTitle></CardHeader>
                    <CardContent className="space-y-3">
                      {(config.header?.custom_links || []).map((link, i) => (
                        <div key={i} className="flex items-center gap-2 p-2 border rounded-lg bg-gray-50">
                          <Input value={link.label} onChange={e => updateHeaderLink(i, 'label', e.target.value)} placeholder="Texte" className="flex-1 h-9" />
                          <Input value={link.url} onChange={e => updateHeaderLink(i, 'url', e.target.value)} placeholder="https://..." className="flex-1 h-9" />
                          <select value={link.style || 'outline'} onChange={e => updateHeaderLink(i, 'style', e.target.value)} className="border rounded px-2 h-9 text-sm">
                            <option value="outline">Contour</option><option value="solid">Plein</option><option value="link">Lien</option></select>
                          <button onClick={() => removeHeaderLink(i)} className="text-red-500 hover:text-red-700 px-2 text-lg font-bold">x</button></div>))}
                      <Button size="sm" variant="outline" onClick={addHeaderLink} data-testid="add-header-link">+ Ajouter un lien</Button>
                    </CardContent></Card>
                  <Card><CardHeader><div className="flex items-center justify-between"><CardTitle className="text-base">Footer</CardTitle>
                    <Toggle label="" small checked={config.footer?.enabled || false} onChange={v => set('footer.enabled', v)} /></div></CardHeader>
                    {config.footer?.enabled && (<CardContent className="space-y-4">
                      <div><label className="text-sm font-medium text-gray-700">Texte</label><Input value={config.footer?.text || ''} onChange={e => set('footer.text', e.target.value)} placeholder="© 2025 Mon Organisation" /></div>
                      <div className="grid grid-cols-2 gap-4">
                        <ColorPicker label="Fond" value={config.footer?.bg_color || '#1f2937'} onChange={v => set('footer.bg_color', v)} />
                        <ColorPicker label="Texte" value={config.footer?.text_color || '#9ca3af'} onChange={v => set('footer.text_color', v)} /></div>
                      <div><label className="text-sm font-medium text-gray-700 mb-2 block">Liens</label>
                        {(config.footer?.links || []).map((l, i) => (<div key={i} className="flex items-center gap-2 mb-2">
                          <Input value={l.label} onChange={e => updateFooterLink(i, 'label', e.target.value)} placeholder="Texte" className="flex-1 h-9" />
                          <Input value={l.url} onChange={e => updateFooterLink(i, 'url', e.target.value)} placeholder="https://..." className="flex-1 h-9" />
                          <button onClick={() => removeFooterLink(i)} className="text-red-500 hover:text-red-700 px-2 text-lg font-bold">x</button></div>))}
                        <Button size="sm" variant="outline" onClick={addFooterLink}>+ Ajouter</Button></div>
                      <div className="rounded-lg p-4 text-center text-sm" style={{ background: config.footer?.bg_color || '#1f2937', color: config.footer?.text_color || '#9ca3af' }}>
                        <p>{config.footer?.text || 'Aperçu'}</p>
                        <div className="flex justify-center gap-4 mt-2">{(config.footer?.links || []).map((l, i) => <span key={i} className="underline">{l.label}</span>)}</div></div>
                    </CardContent>)}</Card>
                </div>
              </TabsContent>

              {/* ===== AVANCÉ ===== */}
              <TabsContent value="advanced">
                <div className="space-y-5">
                  <Card><CardHeader><CardTitle className="text-base">CSS personnalisé</CardTitle></CardHeader>
                    <CardContent><textarea value={config.custom_css || ''} onChange={e => set('custom_css', e.target.value)} className="w-full h-48 font-mono text-sm border rounded-lg p-3 bg-gray-50" placeholder={`.App { background: #000; }`} /></CardContent></Card>
                  <Card><CardHeader><CardTitle className="text-base">Réinitialiser</CardTitle></CardHeader>
                    <CardContent><Button variant="outline" className="border-red-300 text-red-600 hover:bg-red-50" onClick={() => { loadConfig(); toast.info('Rechargé'); }} data-testid="reset-config-btn">Annuler</Button></CardContent></Card>
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
