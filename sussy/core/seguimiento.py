from typing import List, Dict, Any

class TrackerSimple:
    """
    Tracker muy básico que asocia detecciones por cercanía (IoU o distancia).
    Mantiene el ID si el objeto está cerca de donde estaba antes.
    """
    def __init__(self, max_dist: int = 50, max_frames_lost: int = 5):
        self.max_dist = max_dist
        self.max_frames_lost = max_frames_lost
        self.tracks = {}  # id -> {box, frames_lost, clase, score}
        self.next_id = 1

    def actualizar(self, detecciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Predicción simple: asumimos que están donde estaban (podría usar Kalman)
        
        # Asociación
        used_ids = set()
        final_tracks = []

        for det in detecciones:
            cx = (det['x1'] + det['x2']) / 2
            cy = (det['y1'] + det['y2']) / 2
            
            best_id = None
            best_dist = float('inf')

            for tid, track in self.tracks.items():
                if tid in used_ids:
                    continue
                
                tcx = (track['box'][0] + track['box'][2]) / 2
                tcy = (track['box'][1] + track['box'][3]) / 2
                
                dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                
                if dist < self.max_dist and dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                # Update track
                self.tracks[best_id]['box'] = [det['x1'], det['y1'], det['x2'], det['y2']]
                self.tracks[best_id]['frames_lost'] = 0
                self.tracks[best_id]['clase'] = det['clase']
                self.tracks[best_id]['score'] = det['score']
                used_ids.add(best_id)
                
                final_tracks.append({
                    "id": best_id,
                    "box": self.tracks[best_id]['box'],
                    "clase": self.tracks[best_id]['clase'],
                    "score": self.tracks[best_id]['score']
                })
            else:
                # New track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "box": [det['x1'], det['y1'], det['x2'], det['y2']],
                    "frames_lost": 0,
                    "clase": det['clase'],
                    "score": det['score']
                }
                used_ids.add(new_id)
                final_tracks.append({
                    "id": new_id,
                    "box": self.tracks[new_id]['box'],
                    "clase": self.tracks[new_id]['clase'],
                    "score": self.tracks[new_id]['score']
                })

        # Prune lost tracks
        to_delete = []
        for tid in self.tracks:
            if tid not in used_ids:
                self.tracks[tid]['frames_lost'] += 1
                if self.tracks[tid]['frames_lost'] > self.max_frames_lost:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]

        return final_tracks
