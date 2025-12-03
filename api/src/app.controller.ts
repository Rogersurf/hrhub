import { Controller, Get, Render } from '@nestjs/common';
import { MatchingService } from './matching/matching.service';

@Controller()
export class AppController {
  constructor(private readonly matchingService: MatchingService) {}

  @Get()
  @Render('dashboard')
  getDashboard() {
    const candidate = this.matchingService.getCandidateData(0);
    const matches = this.matchingService.findTopMatches(10);
    
    return {
      candidate,
      matches,
      totalMatches: matches.length,
      avgScore: (matches.reduce((sum, m) => sum + m.score, 0) / matches.length).toFixed(3),
      bestScore: matches[0]?.score.toFixed(3) || '0',
    };
  }
}