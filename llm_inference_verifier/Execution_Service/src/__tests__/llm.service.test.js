const { getLLMResponse, truncateText } = require('../llm.service');

describe('LLMService', () => {
  describe('Integration Tests', () => {
    it('should successfully send a query to Hyperbolic LLM and get a response', async () => {
      const query = 'What is the purpose of MotherDAO?';
      const response = await getLLMResponse(query);

      // Print the response for inspection
      console.log('\nLLM Response:', {
        query,
        response: response.choice,
        model: response.model
      });

      // Verify response structure
      expect(response).toHaveProperty('choice');
      expect(response).toHaveProperty('status', 200);
      expect(response).toHaveProperty('model');
      
      // Verify we got a meaningful response
      expect(typeof response.choice).toBe('string');
      expect(response.choice.length).toBeGreaterThan(0);
    }, 30000); // Increase timeout to 30s for API call
  });

  describe('truncateText', () => {
    it('should not truncate text shorter than maxChars', () => {
      const text = 'Hello World';
      expect(truncateText(text, 20)).toBe(text);
    });

    it('should truncate text longer than maxChars', () => {
      const text = 'Hello World';
      expect(truncateText(text, 5)).toBe('Hello');
    });

    it('should handle empty string', () => {
      expect(truncateText('', 10)).toBe('');
    });
  });
});
