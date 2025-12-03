export const mockCandidate = {
  id: 0,
  name: 'Candidate #0',
  skills: ['Python', 'Machine Learning', 'Data Science', 'SQL', 'AWS'],
  experience: '5 years',
  education: 'Master in Computer Science',
  embedding: [0.1, 0.2, 0.3, 0.4, 0.5] // Simplificado (real seria 384 dims)
};

export const mockCompanies = [
  {
    id: 42,
    name: 'Anblicks',
    title: 'Data Scientist',
    embedding: [0.12, 0.19, 0.31, 0.38, 0.52]
  },
  {
    id: 15,
    name: 'iO Associates',
    title: 'ML Engineer',
    embedding: [0.11, 0.21, 0.29, 0.41, 0.48]
  },
  {
    id: 89,
    name: 'DATAECONOMY',
    title: 'Senior Data Analyst',
    embedding: [0.09, 0.18, 0.33, 0.39, 0.51]
  }
];