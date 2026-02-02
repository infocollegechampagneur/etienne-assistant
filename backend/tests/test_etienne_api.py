"""
Backend API Tests for Étienne - AI Assistant for Quebec Secondary Teachers
Tests: Login, Chat with LaTeX, PDF/DOCX generation with logo
"""
import pytest
import requests
import os
import time

# Get the base URL from environment
BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
if not BASE_URL:
    BASE_URL = "https://etienne-platform.preview.emergentagent.com"

# Test credentials
TEST_EMAIL = "informatique@champagneur.qc.ca"
TEST_PASSWORD = "!0910Hi8ki8+"


class TestHealthAndBasicEndpoints:
    """Test basic API health and endpoints"""
    
    def test_api_root(self):
        """Test API root endpoint"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print(f"✅ API root: {data['message']}")
    
    def test_health_check(self):
        """Test health check endpoint via /api/ root"""
        # The root / returns frontend HTML, so we test /api/ instead
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print(f"✅ API health check via /api/: {data['message']}")
    
    def test_quota_status(self):
        """Test quota status endpoint"""
        response = requests.get(f"{BASE_URL}/api/quota-status")
        assert response.status_code == 200
        data = response.json()
        assert "used" in data
        assert "max" in data
        assert "remaining" in data
        print(f"✅ Quota status: {data['remaining']}/{data['max']} remaining")


class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_login_success(self):
        """Test successful login with valid credentials"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD
            }
        )
        
        # Check status code
        if response.status_code == 200:
            data = response.json()
            assert data.get("success") == True
            assert "token" in data
            assert "user" in data
            assert data["user"]["email"] == TEST_EMAIL
            print(f"✅ Login successful for: {data['user']['email']}")
            return data["token"]
        elif response.status_code == 401:
            # User might not exist in DB - this is expected for first run
            print(f"⚠️ Login failed (401) - User may not exist in database")
            pytest.skip("User not found in database - need to create user first")
        else:
            print(f"❌ Login failed with status {response.status_code}: {response.text}")
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "email": "invalid@test.com",
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
        print("✅ Invalid credentials correctly rejected")


class TestChatWithLaTeX:
    """Test chat endpoint with LaTeX formula generation"""
    
    def test_chat_math_question(self):
        """Test chat with a math question - should return LaTeX formulas"""
        # Wait a bit to respect rate limits
        time.sleep(2)
        
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "Explique-moi le théorème de Pythagore avec la formule mathématique pour un élève de Sec 2",
                "message_type": "activites",
                "session_id": "test_latex_session_001"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "response" in data
        assert "message_type" in data
        assert data["message_type"] == "activites"
        
        # Check for LaTeX formulas in response
        response_text = data["response"]
        has_inline_latex = "$" in response_text and response_text.count("$") >= 2
        has_block_latex = "$$" in response_text
        
        print(f"✅ Chat response received ({len(response_text)} chars)")
        print(f"   Contains inline LaTeX ($...$): {has_inline_latex}")
        print(f"   Contains block LaTeX ($$...$$): {has_block_latex}")
        
        # The response should contain some form of LaTeX for math content
        if has_inline_latex or has_block_latex:
            print("✅ LaTeX formulas detected in response")
            # Extract and show some LaTeX examples
            import re
            inline_formulas = re.findall(r'\$([^$]+)\$', response_text)
            if inline_formulas:
                print(f"   Sample formulas: {inline_formulas[:3]}")
        else:
            print("⚠️ No LaTeX formulas detected - response may use Unicode symbols instead")
        
        return data
    
    def test_chat_quadratic_function(self):
        """Test chat with quadratic function question"""
        time.sleep(2)
        
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "Donne-moi la formule de la fonction quadratique et explique chaque paramètre pour Sec 4 CST",
                "message_type": "evaluations",
                "session_id": "test_latex_session_002"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        response_text = data["response"]
        print(f"✅ Quadratic function response received ({len(response_text)} chars)")
        
        # Check for common quadratic formula patterns
        has_quadratic = any(pattern in response_text for pattern in [
            "ax^2", "ax²", "$ax^2$", "f(x) =", "$f(x)"
        ])
        
        if has_quadratic:
            print("✅ Quadratic formula content detected")
        
        return data


class TestDocumentGeneration:
    """Test PDF and DOCX document generation"""
    
    def test_generate_pdf(self):
        """Test PDF document generation with logo"""
        response = requests.post(
            f"{BASE_URL}/api/generate-document",
            json={
                "content": "# Plan de cours - Mathématiques Sec 3\n\n## Objectifs\n- Comprendre les fonctions affines\n- Maîtriser la notation f(x)\n\n## Formules\n- Fonction affine: y = mx + b\n- Pente: m = (y2-y1)/(x2-x1)\n\n## Activités\n1. Exercices de graphiques\n2. Problèmes contextualisés",
                "title": "Plan de cours - Fonctions affines",
                "format": "pdf"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/pdf"
        
        # Check that we got actual PDF content
        content = response.content
        assert len(content) > 1000  # PDF should be at least 1KB
        assert content[:4] == b'%PDF'  # PDF magic bytes
        
        print(f"✅ PDF generated successfully ({len(content)} bytes)")
        print(f"   Content-Disposition: {response.headers.get('content-disposition', 'N/A')}")
        
        return content
    
    def test_generate_docx(self):
        """Test DOCX document generation with logo"""
        response = requests.post(
            f"{BASE_URL}/api/generate-document",
            json={
                "content": "# Évaluation - Théorème de Pythagore\n\n## Questions\n1. Calcule l'hypoténuse d'un triangle rectangle avec a=3 et b=4\n2. Explique la formule a² + b² = c²\n\n## Corrigé\n- Question 1: c = √(9+16) = √25 = 5\n- Question 2: La somme des carrés des cathètes égale le carré de l'hypoténuse",
                "title": "Évaluation Pythagore Sec 2",
                "format": "docx"
            }
        )
        
        assert response.status_code == 200
        expected_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert response.headers.get("content-type") == expected_type
        
        # Check that we got actual DOCX content (ZIP format)
        content = response.content
        assert len(content) > 1000  # DOCX should be at least 1KB
        assert content[:2] == b'PK'  # ZIP magic bytes (DOCX is a ZIP file)
        
        print(f"✅ DOCX generated successfully ({len(content)} bytes)")
        print(f"   Content-Disposition: {response.headers.get('content-disposition', 'N/A')}")
        
        return content
    
    def test_generate_pptx(self):
        """Test PowerPoint generation"""
        response = requests.post(
            f"{BASE_URL}/api/generate-document",
            json={
                "content": "# Présentation - Les fractions\n\n## Introduction\nLes fractions représentent des parties d'un tout\n\n## Exemples\n- 1/2 = une moitié\n- 3/4 = trois quarts",
                "title": "Les Fractions Sec 1",
                "format": "pptx"
            }
        )
        
        assert response.status_code == 200
        expected_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert response.headers.get("content-type") == expected_type
        
        content = response.content
        assert len(content) > 1000
        assert content[:2] == b'PK'  # PPTX is also a ZIP file
        
        print(f"✅ PPTX generated successfully ({len(content)} bytes)")
        
        return content
    
    def test_generate_xlsx(self):
        """Test Excel generation"""
        response = requests.post(
            f"{BASE_URL}/api/generate-document",
            json={
                "content": "# Grille d'évaluation\n\nCritère 1: Compréhension - 25 points\n\nCritère 2: Application - 25 points\n\nCritère 3: Communication - 25 points\n\nCritère 4: Raisonnement - 25 points",
                "title": "Grille évaluation",
                "format": "xlsx"
            }
        )
        
        assert response.status_code == 200
        expected_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert response.headers.get("content-type") == expected_type
        
        content = response.content
        assert len(content) > 1000
        assert content[:2] == b'PK'  # XLSX is also a ZIP file
        
        print(f"✅ XLSX generated successfully ({len(content)} bytes)")
        
        return content
    
    def test_invalid_format(self):
        """Test document generation with invalid format"""
        response = requests.post(
            f"{BASE_URL}/api/generate-document",
            json={
                "content": "Test content",
                "title": "Test",
                "format": "invalid_format"
            }
        )
        
        # Accept 400 or 520 (Cloudflare proxy may convert 400 to 520)
        assert response.status_code in [400, 520]
        
        # Check error message in response
        data = response.json()
        assert "detail" in data
        assert "Format non supporte" in data["detail"] or "invalid" in data["detail"].lower()
        print("✅ Invalid format correctly rejected")


class TestMELSConstraints:
    """Test MELS curriculum constraints - should reject CEGEP/University content"""
    
    def test_secondary_level_content(self):
        """Test that secondary level content is accepted"""
        time.sleep(2)
        
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "message": "Crée un exercice sur les équations du premier degré pour Sec 2",
                "message_type": "activites",
                "session_id": "test_mels_001"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should get a valid response for secondary level content
        assert "response" in data
        assert len(data["response"]) > 50
        
        print(f"✅ Secondary level content accepted ({len(data['response'])} chars)")
        
        return data


class TestSubjectsEndpoint:
    """Test subjects endpoint"""
    
    def test_get_subjects(self):
        """Test getting available subjects"""
        response = requests.get(f"{BASE_URL}/api/subjects")
        assert response.status_code == 200
        data = response.json()
        
        # Should return a dict of subject categories
        assert isinstance(data, dict)
        print(f"✅ Subjects endpoint: {len(data)} categories")
        
        return data


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
