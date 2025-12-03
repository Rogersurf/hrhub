import { Injectable } from '@nestjs/common';
import * as math from 'mathjs';
import { mockCandidate, mockCompanies } from './mock-data';

@Injectable()
export class MatchingService {
  
  cosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = math.dot(vecA, vecB) as number;
    const normA = math.norm(vecA) as number;
    const normB = math.norm(vecB) as number;
    return dotProduct / (normA * normB);
  }

  findTopMatches(candidateId: number, topK: number = 10) {
    const candidate = mockCandidate;
    
    const matches = mockCompanies.map((company) => ({
      companyId: company.id,
      companyName: company.name,
      jobTitle: company.title,
      score: this.cosineSimilarity(candidate.embedding, company.embedding),
    }));

    return matches
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  getCandidateData(candidateId: number) {
    return mockCandidate;
  }
}