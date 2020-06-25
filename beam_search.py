class Beam():
    def __init__(self, init_id=None, hidden=None, log_prob=None, eos_id=None, beam=None):
        if isinstance(beam, Beam):
            self.tokens = [i for i in beam.tokens]
            self.hidden = beam.hidden
            self.log_probs = [i for i in beam.log_probs]
            self.score = beam.score
            self.eos_id = beam.eos_id
            self.done = beam.done
        else:
            if init_id is None:
                raise ValueError("Init id not specified")
            self.tokens = [init_id]
            self.hidden = hidden
            self.log_probs = [log_prob]
            self.score = log_prob 
            self.eos_id = eos_id
            self.done = False

    def forward_beam(self, next_id, hidden, log_prob):
        # return this beam if eos is reached.
        if self.done: return self
        
        # Initialize new beam
        next_beam = Beam(beam=self)
        
        # Update the new beam
        next_beam.tokens.append(next_id)
        next_beam.hidden = hidden
        next_beam.log_probs.append(log_prob)
        
        # Set done flag if <eos> is reached
        if next_id == next_beam.eos_id:
            next_beam.done = True
        return next_beam
    
    def get_score(self):
#         return abs(sum(self.log_probs))
        if len(self.tokens) == 1:
            return abs(sum(self.log_probs))
        return abs(sum(self.log_probs)) / float(len(self.tokens) - 1 + 1e-6)

    def get_state(self):
        return (self.tokens[-1], self.hidden, self.log_probs[-1], self.score)


class BeamQueue():
    def __init__(self, batch_size, beam_size, eos_id):
        # One queue for each batch_item
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.eos_id = eos_id
        self.queues = [[] for i in range(batch_size)]
        self.completed = [False for i in range(batch_size)]
        self.hidden_lstm = False
    
    def add_new(self, item_id, beams):
        queue = self.queues[item_id]
        beams = list(sorted(beams, key=lambda b: b.get_score()))
        for beam in beams:
            queue.append(beam)
            print("add_new: it_id", item_id , [i2w[idx] for idx in beam.tokens], beam.get_score())
        if isinstance(beams[0].hidden, tuple):
            self.hidden_lstm = True
    
    def update_queue(self, item_id, beam_items):
        queue = self.queues[item_id]
        print("queue_id:" , item_id, " len beam_items", len(beam_items))

        new_beams = []
        item_no = 0
        for i in range(len(queue)):
            beam = queue[i]
            for j in range(self.beam_size):
                token_id, hidden, log_prob = beam_items[item_no]
                candidate_beam = beam.forward_beam(token_id, hidden, log_prob)
                new_beams.append(candidate_beam)
                print([i2w[idx] for idx in candidate_beam.tokens], candidate_beam.get_score())
                item_no += 1

        beams = list(sorted(new_beams, key=lambda b: b.get_score()))[: self.beam_size]
        print("After update len beam:", len(beams))
        
        done = True
        for beam in beams:
            if beam.tokens[-1] != self.eos_id:
                done = False
                break

        self.queues[item_id] = beams
        return done

    def get_new_iter_data(self):
        """
        Returns:
            1. output_ids: the previous token ids.
                shape: (batch_size, 1)
            2. hidden: the previous hidden state for each beam
                a) GRU, RNN: one tensor element
                b) LSTM: Tuple of hidden and cell_states
        """
        
        output_ids_shape = (self.batch_size)
        if self.hidden_lstm:
            hidden_shape = (self.queues[0][0].hidden[0].size(0), self.batch_size, self.queues[0][0].hidden[0].size(2))
        else:
            hidden_shape = (self.queues[0][0].hidden.size(0), self.batch_size, self.queues[0][0].hidden.size(2))
        
        for beam_no in range(self.beam_size):
            output_ids = torch.zeros(output_ids_shape).long().to(DEVICE)
            
            if self.hidden_lstm:
                hidden = [[], []]
            else:
                hidden = []
                
            for batch_no in range(self.batch_size):
                beam = self.queues[batch_no][beam_no]
                output_ids[batch_no] = beam.tokens[-1]
                if self.hidden_lstm:
                    hidden[0].append(beam.hidden[0])
                    hidden[1].append(beam.hidden[1])
                else:
#                     hidden[:, batch_no, :] = beam.hidden[:, 0, :]
                     hidden.append(beam.hidden)
            
            if self.hidden_lstm:
                hidden[0] = torch.cat(hidden[0], dim=1)
                hidden[1] = torch.cat(hidden[1], dim=1)
            else:
                hidden = torch.cat(hidden, dim=1)
            
            yield output_ids, hidden

def decode_beam(decoder, input_ids, input_lens, hidden, eos_id, decoder_func, beam_size=5, batch_first=True):
    """
      1. decoder: The decoder module with forward implemented
      2. input_embeddings: (Batch_size, timesteps) # Should be atleast 1 (<SOS> token)
      3. hidden: encoder's last hidden state
      4. eos_id: number which maps to eos token (needed for termination of beam)
      5. decoder_func: run one step of decoder
      8. beam_size: width of beam search
      9. batch_first: same as decoder's batch_first
    """
    if batch_first:
        batch_size, num_timesteps = input_ids.size()
    else:
        raise ValueError("Not Implemented")
    
    print(input_ids.shape, input_ids)
    for t in range(num_timesteps):
        output, hidden = decoder_func(input_ids[:, t], hidden, input_lens)
    
    # Define Vars to use
    is_lstm = isinstance(hidden, tuple)
    outputs = []
    beams = BeamQueue(batch_size, beam_size, eos_id)
    
    # Select top beam_size elements
    output_vals, output_ids = torch.topk(output, beam_size, dim=-1) # (batch_size, beam_size), (batch_size, beam_size)
    print("Top elems: ", [[i2w[i.item()] for i in j] for j in output_ids])


    # Add to the BeamQueue
    for i in range(batch_size):
        # For each validation sample 
        beam_list = []
        for j in range(beam_size):
            if is_lstm:
                beam = Beam(output_ids[i][j].item(), (hidden[0][:, i, :].unsqueeze(1), hidden[1][:, i, :].unsqueeze(1)), output_vals[i][j].item(), eos_id)
            else:
                beam = Beam(output_ids[i][j].item(), hidden[:, i, :].unsqueeze(1), output_vals[i][j].item(), eos_id)
            beam_list.append(beam)
            
        # Add the new beams to the beam queue 
        beams.add_new(i, beam_list)
    
    while True:
        output_ids = []
        output_vals = []
        hiddens = [[] for _ in range(batch_size)]
        
        for input_ids, hidden in beams.get_new_iter_data():
            print("ip ids", [i2w[i.item()] for i in input_ids], " hidden", hidden[0].shape)
            output, hidden = decoder_func(input_ids, hidden, input_lens)
        
            # (batch_size * beam_size, beam_size), (batch_size * beam_size, beam_size)
            op_vals, op_ids = torch.topk(output, beam_size, dim=-1)
            output_vals.append(op_vals)
            output_ids.append(op_ids)
            for bno in range(batch_size):
                if is_lstm:
                    hiddens[bno].append((hidden[0][:, bno, :].unsqueeze(1), hidden[1][:, bno, :].unsqueeze(1)))
                else:
                    hiddens[bno].append(hidden[:, bno, :].unsqueeze(1))

        output_vals = torch.cat(output_vals, dim=1)
        output_ids = torch.cat(output_ids, dim=1)
        
        print("op1 ids", [i2w[i.item()] for i in output_ids[0]])
        print("op2 ids", [i2w[i.item()] for i in output_ids[1]])
        print(output_vals.shape)
        
        beams_done = 0
        for batch_no in range(batch_size):
            beam_items = []
            for j in range(beam_size * beam_size):
                item = [output_ids[batch_no][j].item(), hiddens[batch_no][j // beam_size], output_vals[batch_no][j].item()]
                beam_items.append(item)
            if beams.update_queue(batch_no, beam_items) == True:
                beams_done += 1
        print("beams_done: ", beams_done)
        if beams_done == batch_size:
            break
    return beams
