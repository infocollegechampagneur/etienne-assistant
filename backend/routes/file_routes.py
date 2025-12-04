"""Routes optimisées pour l'upload et l'analyse de fichiers"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import asyncio
import logging
from io import BytesIO

# Router
router = APIRouter(tags=["Files"])

# MongoDB connection (sera injecté depuis server.py)
db = None

def set_db(database):
    """Configure la connexion à la base de données"""
    global db
    db = database

@router.post("/api/upload-files-batch")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload et extraction de plusieurs fichiers en parallèle (OPTIMISÉ)
    
    Cette route traite plusieurs fichiers en parallèle pour plus de rapidité.
    Maximum 5 fichiers, 10MB par fichier.
    """
    try:
        max_files = 5
        max_size = 10 * 1024 * 1024  # 10MB
        
        if len(files) > max_files:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_files} fichiers autorisés"
            )
        
        # Import de la fonction d'extraction
        from server import extract_text_from_file
        
        async def process_single_file(file: UploadFile, index: int):
            """Traite un seul fichier"""
            try:
                # Lire le contenu
                content = await file.read()
                file_size = len(content)
                
                if file_size > max_size:
                    return {
                        "success": False,
                        "filename": file.filename,
                        "error": f"Fichier trop volumineux ({file_size / 1024 / 1024:.1f}MB > 10MB)"
                    }
                
                # Remettre le pointeur au début
                file.file = BytesIO(content)
                
                # Extraire le texte
                extracted_text = await extract_text_from_file(file)
                
                # Limiter la longueur
                max_text_length = 8000  # Légèrement moins pour permettre plusieurs fichiers
                if len(extracted_text) > max_text_length:
                    extracted_text = extracted_text[:max_text_length] + "\n\n[...Texte tronqué...]"
                
                return {
                    "success": True,
                    "filename": file.filename,
                    "file_size": file_size,
                    "extracted_text": extracted_text,
                    "text_length": len(extracted_text),
                    "index": index
                }
                
            except Exception as e:
                logging.error(f"Erreur traitement {file.filename}: {e}")
                return {
                    "success": False,
                    "filename": file.filename,
                    "error": str(e)
                }
        
        # Traiter tous les fichiers en parallèle
        tasks = [process_single_file(file, i) for i, file in enumerate(files)]
        results = await asyncio.gather(*tasks)
        
        # Séparer succès et échecs
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        if len(successful) == 0:
            raise HTTPException(
                status_code=500,
                detail="Aucun fichier n'a pu être traité"
            )
        
        # Combiner tous les textes extraits
        combined_text = ""
        for result in successful:
            combined_text += f"\n\n=== DOCUMENT {result['index'] + 1}: {result['filename']} ===\n"
            combined_text += result['extracted_text']
        
        return {
            "success": True,
            "files_processed": len(successful),
            "files_failed": len(failed),
            "total_size": sum(r["file_size"] for r in successful),
            "combined_text": combined_text.strip(),
            "combined_length": len(combined_text),
            "files": successful,
            "errors": failed if failed else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur upload batch: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )
