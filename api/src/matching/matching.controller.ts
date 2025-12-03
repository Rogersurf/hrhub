import { Controller, Get, Query, Param } from '@nestjs/common';
import { MatchingService } from './matching.service';

@Controller('api/matching')
export class MatchingController {
  constructor(private readonly matchingService: MatchingService) {}

  @Get('candidate/:id/matches')
  getMatches(
    @Param('id') candidateId: string,
    @Query('topK') topK?: number
  ) {
    const id = parseInt(candidateId);
    const k = topK ? parseInt(topK.toString()) : 10;
    
    return {
      candidate: this.matchingService.getCandidateData(id),
      matches: this.matchingService.findTopMatches(k),
    };
  }
}