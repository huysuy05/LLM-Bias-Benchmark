import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class RobustInferenceOptimizer:
    def __init__(self, model, tokenizer, D_pref, lambda_reg=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.D_pref = D_pref  # list of (text, label) pairs
        self.lambda_reg = lambda_reg
        self.device = model.device
        
    def get_embeddings(self, texts):
        """Get embeddings for texts using the model"""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state mean as embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings.cpu().numpy()
    
    def create_perturbations(self, text, num_perturbations=5):
        """Create semantic perturbations of input text"""
        perturbations = []
        
        # Simple paraphrasing strategies (in practice, use more sophisticated methods)
        words = text.split()
        
        # 1. Synonym replacement (placeholder - use real synonym library)
        if len(words) > 3:
            perturbed = words.copy()
            # Simple: reverse some words to simulate change
            if len(perturbed) > 4:
                perturbed[1], perturbed[2] = perturbed[2], perturbed[1]
            perturbations.append(' '.join(perturbed))
        
        # 2. Add context
        contexts = [
            "Considering different perspectives, ",
            "In various contexts, ",
            "From another viewpoint, "
        ]
        for context in contexts[:2]:
            perturbations.append(context + text)
            
        # 3. Question reformulation (if it's a question)
        if "?" in text:
            reforms = [
                text.replace("?", " - let's think about this carefully."),
                "I need to understand: " + text.replace("?", "")
            ]
            perturbations.extend(reforms)
            
        # Ensure we have exactly num_perturbations
        while len(perturbations) < num_perturbations:
            perturbations.append(text)  # Keep original as one perturbation
            
        return perturbations[:num_perturbations]
    
    def compute_similarity_matrix(self, perturbations, pref_texts):
        """Compute cosine similarity between perturbations and preference texts"""
        all_texts = perturbations + pref_texts
        embeddings = self.get_embeddings(all_texts)
        
        pert_embeddings = embeddings[:len(perturbations)]
        pref_embeddings = embeddings[len(perturbations):]
        
        # Normalize embeddings
        pert_embeddings = pert_embeddings / np.linalg.norm(pert_embeddings, axis=1, keepdims=True)
        pref_embeddings = pref_embeddings / np.linalg.norm(pref_embeddings, axis=1, keepdims=True)
        
        # Cosine similarity matrix
        similarity_matrix = np.dot(pert_embeddings, pref_embeddings.T)
        return similarity_matrix
    
    def get_model_probabilities(self, texts, possible_answers):
        """Get model probabilities for possible answers given input texts"""
        probs_matrix = []
        
        for text in texts:
            text_probs = []
            for answer in possible_answers:
                # Create prompt: "Text: {text}\nAnswer: {answer}"
                prompt = f"{text}\nAnswer: {answer}"
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    prob = F.softmax(logits, dim=-1)
                    
                    # Get probability of the answer token
                    answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                    if len(answer_tokens) > 0:
                        answer_prob = prob[answer_tokens[0]].item()
                    else:
                        answer_prob = 0.0
                    text_probs.append(answer_prob)
            
            # Normalize probabilities for this text
            text_probs = np.array(text_probs)
            if text_probs.sum() > 0:
                text_probs = text_probs / text_probs.sum()
            probs_matrix.append(text_probs)
        
        return np.array(probs_matrix)
    
    def project_to_simplex(self, w):
        """Project weights to probability simplex"""
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, len(w) + 1)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        return np.maximum(w - theta, 0)
    
    def objective_function(self, w, S, P, lambda_reg):
        """Compute the robust objective function value"""
        # P_robust = sum_over_perturbations [ sum_over_pref (w_i * S * P) ]
        robust_probs = np.zeros(P.shape[1])  # probabilities for each possible answer
        
        for pert_idx in range(P.shape[0]):
            for pref_idx in range(len(w)):
                robust_probs += w[pref_idx] * S[pert_idx, pref_idx] * P[pert_idx, :]
        
        # Log probability of most likely answer
        max_prob = np.max(robust_probs)
        if max_prob <= 0:
            log_prob = -np.inf
        else:
            log_prob = np.log(max_prob)
        
        # KL regularization
        uniform = np.ones_like(w) / len(w)
        kl_penalty = np.sum(w * np.log(w / uniform))
        
        return log_prob - lambda_reg * kl_penalty
    
    def solve_optimization(self, x, possible_answers, num_iterations=100, learning_rate=0.1):
        """Main optimization routine"""
        
        # Step 1: Create perturbations
        perturbations = self.create_perturbations(x)
        pref_texts = [item[0] for item in self.D_pref]
        
        print(f"Created {len(perturbations)} perturbations")
        print(f"Using {len(pref_texts)} preference examples")
        
        # Step 2: Compute similarity matrix
        S = self.compute_similarity_matrix(perturbations, pref_texts)
        print(f"Similarity matrix shape: {S.shape}")
        
        # Step 3: Get model probabilities
        P = self.get_model_probabilities(perturbations, possible_answers)
        print(f"Probability matrix shape: {P.shape}")
        
        # Step 4: Initialize weights uniformly
        w = np.ones(len(self.D_pref)) / len(self.D_pref)
        
        # Step 5: Gradient ascent
        best_w = w.copy()
        best_objective = -np.inf
        
        for iteration in range(num_iterations):
            # Compute gradient numerically (more stable than analytical for this complex function)
            current_obj = self.objective_function(w, S, P, self.lambda_reg)
            
            if current_obj > best_objective:
                best_objective = current_obj
                best_w = w.copy()
            
            # Finite difference gradient
            grad = np.zeros_like(w)
            epsilon = 1e-8
            
            for i in range(len(w)):
                w_plus = w.copy()
                w_plus[i] += epsilon
                w_plus = self.project_to_simplex(w_plus)
                
                obj_plus = self.objective_function(w_plus, S, P, self.lambda_reg)
                grad[i] = (obj_plus - current_obj) / epsilon
            
            # Gradient ascent step
            w_new = w + learning_rate * grad
            w = self.project_to_simplex(w_new)
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Objective = {current_obj:.4f}")
        
        print(f"Final weights: {best_w}")
        return best_w, S, P, perturbations
    
    def predict(self, x, possible_answers):
        """Make robust prediction for input x"""
        w_opt, S, P, perturbations = self.solve_optimization(x, possible_answers)
        
        # Compute final robust probabilities
        robust_probs = np.zeros(len(possible_answers))
        
        for pert_idx in range(P.shape[0]):
            for pref_idx in range(len(w_opt)):
                robust_probs += w_opt[pref_idx] * S[pert_idx, pref_idx] * P[pert_idx, :]
        
        # Select best answer
        best_idx = np.argmax(robust_probs)
        best_answer = possible_answers[best_idx]
        confidence = robust_probs[best_idx]
        
        return best_answer, confidence, robust_probs