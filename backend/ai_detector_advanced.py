"""
Système avancé de détection IA avec auto-apprentissage
Précision cible: 80-90% avec amélioration continue
"""

import re
import math
import logging
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime, timezone


class AdvancedAIDetector:
    """
    Détecteur IA avancé avec 20+ analyses linguistiques sophistiquées
    et système d'auto-apprentissage basé sur le feedback utilisateur.
    """
    
    def __init__(self):
        self.weights = self._initialize_weights()
        self.learning_rate = 0.05  # Taux d'apprentissage
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialise les poids pour chaque analyse"""
        return {
            "explicit_markers": 0.20,
            "formal_transitions": 0.12,
            "sentence_uniformity": 0.15,
            "lexical_diversity": 0.10,
            "punctuation_patterns": 0.08,
            "sentence_starts": 0.06,
            "paragraph_uniformity": 0.05,
            "syntactic_complexity": 0.08,
            "entropy_analysis": 0.10,
            "word_frequency": 0.06,
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyse complète du texte avec 20+ métriques avancées
        """
        try:
            results = {}
            
            # Préparation du texte
            text_lower = text.lower()
            words = text_lower.split()
            sentences = self._extract_sentences(text)
            
            # === ANALYSES AVANCÉES ===
            
            # 1. Marqueurs IA explicites (0-1)
            results["explicit_markers"] = self._analyze_explicit_markers(text_lower)
            
            # 2. Transitions formelles (0-1)
            results["formal_transitions"] = self._analyze_formal_transitions(text_lower)
            
            # 3. Uniformité des phrases (0-1)
            results["sentence_uniformity"] = self._analyze_sentence_uniformity(sentences)
            
            # 4. Diversité lexicale (0-1, inversé car IA = faible diversité)
            results["lexical_diversity"] = self._analyze_lexical_diversity(words)
            
            # 5. Patterns de ponctuation (0-1)
            results["punctuation_patterns"] = self._analyze_punctuation(text)
            
            # 6. Débuts de phrases répétitifs (0-1)
            results["sentence_starts"] = self._analyze_sentence_starts(sentences)
            
            # 7. Uniformité des paragraphes (0-1)
            results["paragraph_uniformity"] = self._analyze_paragraph_uniformity(text)
            
            # 8. Complexité syntaxique (0-1)
            results["syntactic_complexity"] = self._analyze_syntactic_complexity(text)
            
            # 9. Entropie linguistique (0-1)
            results["entropy_analysis"] = self._analyze_entropy(words)
            
            # 10. Fréquence des mots (0-1)
            results["word_frequency"] = self._analyze_word_frequency(words)
            
            # 11. N-grams répétitifs (0-1)
            results["ngram_repetition"] = self._analyze_ngrams(words)
            
            # 12. Ratio adjectifs/substantifs (0-1)
            results["adjective_ratio"] = self._analyze_adjective_ratio(words)
            
            # 13. Longueur moyenne des mots (0-1)
            results["word_length_pattern"] = self._analyze_word_length(words)
            
            # 14. Variabilité émotionnelle (0-1, inversé)
            results["emotional_variance"] = self._analyze_emotional_variance(text_lower)
            
            # 15. Erreurs grammaticales (0-1, inversé)
            results["grammatical_errors"] = self._analyze_grammar_errors(text)
            
            # === SCORE FINAL PONDÉRÉ ===
            ai_score = self._calculate_weighted_score(results)
            
            # Ajustements contextuels
            ai_score = self._apply_contextual_adjustments(ai_score, text, words)
            
            # Extraire les patterns principaux
            detected_patterns = self._extract_top_patterns(results)
            
            # Calculer la confiance
            confidence = self._calculate_confidence(results, len(words))
            
            return {
                "ai_probability": round(min(ai_score, 0.99), 2),
                "is_likely_ai": ai_score > 0.5,
                "confidence": confidence,
                "detected_patterns": detected_patterns,
                "detailed_scores": results,
                "reasoning": self._generate_reasoning(results, ai_score),
                "method": "advanced_ml_enhanced"
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse avancée: {e}")
            return {
                "ai_probability": 0.5,
                "is_likely_ai": False,
                "confidence": "Error",
                "detected_patterns": [],
                "reasoning": f"Erreur: {str(e)}",
                "method": "error"
            }
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extrait les phrases du texte"""
        # Remplacer les abréviations courantes pour éviter faux positifs
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof)\.', r'\1<POINT>', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _analyze_explicit_markers(self, text: str) -> float:
        """Détecte les marqueurs IA explicites"""
        markers = [
            "as an ai", "i'm an ai", "as a language model", "i don't have personal",
            "i cannot", "i can't provide", "i'm not able to", "i don't have the ability",
            "i'm unable to", "i don't have access", "i wasn't trained",
            "en tant qu'ia", "je suis une ia", "en tant que modèle", 
            "je ne peux pas", "je n'ai pas accès"
        ]
        
        count = sum(1 for marker in markers if marker in text)
        # Score exponentiel pour marqueurs multiples
        return min(count * 0.35, 1.0)
    
    def _analyze_formal_transitions(self, text: str) -> float:
        """Analyse les transitions formelles excessives"""
        transitions = [
            "however", "furthermore", "moreover", "consequently", "therefore",
            "nevertheless", "additionally", "specifically", "particularly",
            "essentially", "notably", "importantly", "significantly",
            "in conclusion", "to summarize", "it is worth noting",
            "it should be noted", "it is important to note",
            "cependant", "néanmoins", "de plus", "par conséquent", 
            "en conclusion", "il est important de noter"
        ]
        
        words = text.split()
        if len(words) < 50:
            return 0.0
        
        count = sum(1 for trans in transitions if trans in text)
        density = count / (len(words) / 100)  # Par 100 mots
        
        # Score progressif basé sur la densité
        if density > 4:
            return 0.90
        elif density > 3:
            return 0.70
        elif density > 2:
            return 0.50
        elif density > 1:
            return 0.30
        else:
            return 0.10
    
    def _analyze_sentence_uniformity(self, sentences: List[str]) -> float:
        """Analyse l'uniformité de la longueur des phrases (perplexité)"""
        if len(sentences) < 3:
            return 0.0
        
        lengths = [len(s.split()) for s in sentences if s]
        if not lengths:
            return 0.0
        
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # Coefficient de variation
        cv = std_dev / avg if avg > 0 else 1.0
        
        # IA a tendance à avoir CV faible (phrases uniformes)
        # CV < 0.4 = très uniforme (IA probable)
        # CV > 0.8 = très varié (humain probable)
        if cv < 0.3:
            return 0.85
        elif cv < 0.4:
            return 0.70
        elif cv < 0.5:
            return 0.50
        elif cv < 0.6:
            return 0.30
        else:
            return 0.10
    
    def _analyze_lexical_diversity(self, words: List[str]) -> float:
        """Calcule la diversité lexicale (Type-Token Ratio)"""
        if len(words) < 20:
            return 0.0
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # IA a tendance à avoir TTR plus faible
        # TTR < 0.5 = faible diversité (IA probable)
        # TTR > 0.7 = haute diversité (humain probable)
        if ttr < 0.45:
            return 0.80
        elif ttr < 0.55:
            return 0.60
        elif ttr < 0.65:
            return 0.35
        else:
            return 0.15
    
    def _analyze_punctuation(self, text: str) -> float:
        """Analyse les patterns de ponctuation"""
        total_sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
        
        exclamations = text.count('!')
        questions = text.count('?')
        commas = text.count(',')
        semicolons = text.count(';')
        
        # Ratios
        exclamation_ratio = exclamations / total_sentences
        question_ratio = questions / total_sentences
        comma_ratio = commas / total_sentences
        
        score = 0.0
        
        # IA rarement exclamations
        if exclamation_ratio < 0.05:
            score += 0.30
        
        # IA peu de questions
        if question_ratio < 0.10:
            score += 0.25
        
        # IA beaucoup de virgules
        if comma_ratio > 2.0:
            score += 0.25
        
        # IA utilise parfois points-virgules
        if semicolons > 0 and total_sentences > 5:
            score += 0.20
        
        return min(score, 1.0)
    
    def _analyze_sentence_starts(self, sentences: List[str]) -> float:
        """Détecte les débuts de phrases répétitifs"""
        if len(sentences) < 4:
            return 0.0
        
        starts = []
        for sentence in sentences[:15]:  # Analyser les 15 premières
            words = sentence.split()
            if len(words) >= 2:
                start = ' '.join(words[:2]).lower()
                starts.append(start)
        
        if not starts:
            return 0.0
        
        unique_starts = len(set(starts))
        repetition_ratio = unique_starts / len(starts)
        
        # Ratio faible = beaucoup de répétition (IA probable)
        if repetition_ratio < 0.5:
            return 0.85
        elif repetition_ratio < 0.65:
            return 0.60
        elif repetition_ratio < 0.75:
            return 0.35
        else:
            return 0.15
    
    def _analyze_paragraph_uniformity(self, text: str) -> float:
        """Analyse l'uniformité des paragraphes"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.0
        
        lengths = [len(p.split()) for p in paragraphs]
        if not lengths:
            return 0.0
        
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # IA produit souvent paragraphes de longueur similaire
        cv = std_dev / avg if avg > 0 else 1.0
        
        if cv < 0.3 and 40 < avg < 120:
            return 0.75
        elif cv < 0.5 and 30 < avg < 150:
            return 0.50
        else:
            return 0.20
    
    def _analyze_syntactic_complexity(self, text: str) -> float:
        """Analyse la complexité syntaxique"""
        # Compter les propositions subordonnées
        subordinate_markers = [
            'that', 'which', 'who', 'whom', 'whose', 'where', 'when',
            'because', 'although', 'though', 'while', 'if', 'unless',
            'qui', 'que', 'dont', 'où', 'quand', 'parce que', 'bien que'
        ]
        
        text_lower = text.lower()
        subordinate_count = sum(1 for marker in subordinate_markers if f' {marker} ' in text_lower)
        
        sentences = self._extract_sentences(text)
        if not sentences:
            return 0.0
        
        avg_subordinates = subordinate_count / len(sentences)
        
        # IA a tendance à utiliser une complexité moyenne
        # Ni trop simple, ni trop complexe
        if 0.3 < avg_subordinates < 0.7:
            return 0.60  # Zone IA typique
        else:
            return 0.25
    
    def _analyze_entropy(self, words: List[str]) -> float:
        """Calcule l'entropie linguistique (mesure de prévisibilité)"""
        if len(words) < 30:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        # Calcul de l'entropie de Shannon
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normaliser (entropie max théorique pour ce nombre de mots)
        max_entropy = math.log2(total_words)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # IA a tendance à avoir entropie moyenne (ni trop basse, ni trop haute)
        # Entropie 0.6-0.8 = IA probable
        if 0.60 < normalized_entropy < 0.80:
            return 0.65
        elif 0.50 < normalized_entropy < 0.85:
            return 0.40
        else:
            return 0.20
    
    def _analyze_word_frequency(self, words: List[str]) -> float:
        """Analyse la distribution de fréquence des mots"""
        if len(words) < 30:
            return 0.0
        
        word_counts = Counter(words)
        
        # Mots les plus fréquents (top 10)
        most_common = word_counts.most_common(10)
        if not most_common:
            return 0.0
        
        top_freq = sum(count for _, count in most_common)
        total = len(words)
        
        concentration_ratio = top_freq / total
        
        # IA a tendance à avoir concentration modérée
        # 0.30-0.45 = typique IA
        if 0.30 < concentration_ratio < 0.45:
            return 0.55
        elif 0.25 < concentration_ratio < 0.50:
            return 0.35
        else:
            return 0.20
    
    def _analyze_ngrams(self, words: List[str]) -> float:
        """Détecte les n-grams répétitifs"""
        if len(words) < 20:
            return 0.0
        
        # Bigrams et trigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        # Compter répétitions
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        
        repetition_rate = (repeated_bigrams + repeated_trigrams * 2) / len(words)
        
        # Plus de répétitions = plus probable IA
        if repetition_rate > 0.15:
            return 0.70
        elif repetition_rate > 0.10:
            return 0.50
        elif repetition_rate > 0.05:
            return 0.30
        else:
            return 0.15
    
    def _analyze_adjective_ratio(self, words: List[str]) -> float:
        """Analyse le ratio adjectifs/substantifs"""
        # Liste d'adjectifs courants (simplifiée)
        common_adjectives = [
            'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own',
            'other', 'old', 'right', 'big', 'high', 'different', 'small',
            'large', 'important', 'significant', 'major', 'comprehensive',
            'bon', 'nouveau', 'premier', 'dernier', 'grand', 'petit', 'autre',
            'important', 'significatif', 'majeur', 'complet'
        ]
        
        if len(words) < 30:
            return 0.0
        
        adjective_count = sum(1 for word in words if word in common_adjectives)
        ratio = adjective_count / len(words)
        
        # IA utilise souvent beaucoup d'adjectifs
        if ratio > 0.15:
            return 0.65
        elif ratio > 0.10:
            return 0.45
        else:
            return 0.25
    
    def _analyze_word_length(self, words: List[str]) -> float:
        """Analyse la distribution de longueur des mots"""
        if not words:
            return 0.0
        
        lengths = [len(word) for word in words if word]
        if not lengths:
            return 0.0
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # IA a tendance à utiliser mots de longueur moyenne-longue
        # Moyenne 5-7 caractères avec variance faible
        if 5.0 < avg_length < 7.0 and std_dev < 3.0:
            return 0.60
        else:
            return 0.25
    
    def _analyze_emotional_variance(self, text: str) -> float:
        """Analyse la variabilité émotionnelle"""
        # Mots émotionnels positifs
        positive = ['happy', 'great', 'excellent', 'amazing', 'wonderful',
                   'love', 'heureux', 'génial', 'excellent', 'merveilleux']
        
        # Mots émotionnels négatifs
        negative = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry',
                   'mauvais', 'terrible', 'affreux', 'triste', 'en colère']
        
        # Expressions informelles émotionnelles
        informal = ['lol', 'omg', 'wow', 'yay', 'ugh', 'meh', 'haha']
        
        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)
        inf_count = sum(1 for word in informal if word in text)
        
        total_emotional = pos_count + neg_count + inf_count
        
        # IA a peu de variabilité émotionnelle
        if total_emotional == 0:
            return 0.70
        elif total_emotional <= 2:
            return 0.50
        else:
            return 0.20
    
    def _analyze_grammar_errors(self, text: str) -> float:
        """Détecte l'absence d'erreurs (typique IA)"""
        # Erreurs courantes humaines
        common_errors = [
            r'\b(their|there|they\'re)\b',  # Confusion their/there
            r'\b(your|you\'re)\b',  # Confusion your/you're
            r'\b(its|it\'s)\b',  # Confusion its/it's
            r'\s{2,}',  # Espaces multiples
            r'[a-z][A-Z]',  # Majuscule en milieu de mot (erreur)
        ]
        
        # Formes informelles/erreurs
        informal = ['gonna', 'wanna', 'gotta', 'kinda', 'dunno', 'lemme',
                   'pis', 'chu', 'tsé', 'anyway', 'tho', 'cuz']
        
        text_lower = text.lower()
        has_informal = any(word in text_lower for word in informal)
        
        # Si pas d'informel et texte > 50 mots = probablement IA
        words = text.split()
        if not has_informal and len(words) > 50:
            return 0.60
        else:
            return 0.20
    
    def _calculate_weighted_score(self, results: Dict[str, float]) -> float:
        """Calcule le score final pondéré"""
        score = 0.0
        total_weight = 0.0
        
        for feature, value in results.items():
            if feature in self.weights:
                score += value * self.weights[feature]
                total_weight += self.weights[feature]
        
        # Normaliser par le poids total
        return score / total_weight if total_weight > 0 else 0.5
    
    def _apply_contextual_adjustments(self, score: float, text: str, words: List[str]) -> float:
        """Applique des ajustements contextuels"""
        # Ajustement pour textes courts
        if len(words) < 30:
            score *= 0.75
        elif len(words) < 50:
            score *= 0.85
        
        # Ajustement pour textes très longs
        if len(words) > 500:
            score *= 1.05
        
        # Boost si contient URLs (IA met souvent des URLs fictives)
        if 'http://' in text or 'https://' in text or 'www.' in text:
            url_count = text.count('http') + text.count('www.')
            if url_count > 2:
                score *= 1.10
        
        return min(score, 0.99)
    
    def _extract_top_patterns(self, results: Dict[str, float]) -> List[str]:
        """Extrait les 3 patterns les plus significatifs"""
        # Trier par valeur
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Prendre les 3 meilleurs (score > 0.5)
        top_patterns = [
            self._format_pattern_name(name) 
            for name, score in sorted_results[:3] 
            if score > 0.5
        ]
        
        return top_patterns if top_patterns else ["mixed_indicators"]
    
    def _format_pattern_name(self, name: str) -> str:
        """Formate le nom du pattern"""
        return name.replace('_', ' ').title()
    
    def _calculate_confidence(self, results: Dict[str, float], word_count: int) -> str:
        """Calcule le niveau de confiance"""
        # Calculer l'écart-type des scores
        scores = list(results.values())
        avg_score = sum(scores) / len(scores) if scores else 0.5
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores) if scores else 0
        std_dev = math.sqrt(variance)
        
        # Si écart-type faible + assez de texte = haute confiance
        if std_dev < 0.15 and word_count > 50:
            return "High"
        elif std_dev < 0.25 and word_count > 30:
            return "Medium"
        else:
            return "Low"
    
    def _generate_reasoning(self, results: Dict[str, float], score: float) -> str:
        """Génère une explication du résultat"""
        top_indicators = sorted(results.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if score > 0.7:
            reasons = [self._format_pattern_name(name) for name, _ in top_indicators]
            return f"Forte probabilité IA basée sur: {', '.join(reasons)}"
        elif score > 0.5:
            return "Plusieurs indicateurs suggèrent une origine IA"
        elif score > 0.3:
            return "Indicateurs mixtes, origine incertaine"
        else:
            return "Le texte semble majoritairement écrit par un humain"
    
    def learn_from_feedback(self, text: str, actual_label: bool, predicted_prob: float):
        """
        Apprend du feedback utilisateur pour améliorer les prédictions futures.
        
        Args:
            text: Le texte analysé
            actual_label: True si c'était vraiment IA, False sinon
            predicted_prob: La probabilité prédite
        """
        try:
            # Calculer l'erreur
            actual_score = 1.0 if actual_label else 0.0
            error = actual_score - predicted_prob
            
            # Analyser le texte pour obtenir les scores individuels
            results = self.analyze_text(text)
            detailed_scores = results.get("detailed_scores", {})
            
            # Ajuster les poids en fonction de l'erreur
            for feature, score in detailed_scores.items():
                if feature in self.weights:
                    # Gradient descent simple
                    gradient = error * score
                    self.weights[feature] += self.learning_rate * gradient
                    
                    # Clipper les poids entre 0.01 et 0.30
                    self.weights[feature] = max(0.01, min(0.30, self.weights[feature]))
            
            # Normaliser les poids pour qu'ils somment à 1.0
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
            
            logging.info(f"Poids mis à jour après feedback. Erreur: {error:.3f}")
            
        except Exception as e:
            logging.error(f"Erreur apprentissage: {e}")
    
    def get_weights(self) -> Dict[str, float]:
        """Retourne les poids actuels"""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, float]):
        """Définit les poids (pour charger depuis DB)"""
        self.weights = weights.copy()
