from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

import os
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

prompt1=PromptTemplate(
    template="Generate Short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="Generate 5 short questions answer from the follwoing text \n {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser=StrOutputParser()


paralle_chain=RunnableParallel({
    "notes":prompt1 | model | parser,
    "quiz":prompt2 | model | parser
})

merge_chain=prompt3 | model | parser
chain= paralle_chain | merge_chain 
text="""
The problem is that our experiential happiness is only loosely correlated with how happy we feel when thinking broadly about our lives.

Kahneman gives the example of a two week vacation. Assuming the vacation was equally enjoyable in every moment, then a two week vacation should be twice as good as a one week vacation. After all, there are twice as many moments of experiencing happiness.

Experiencing two weeks on vacation
However, from the standpoint of memory, a two-week vacation is barely better than a one-week vacation. This is because there are no new memories added in this time, so all the similar moments of happiness are simply forgotten.

Memory doesn't count the minutes
Here we have the conflict. Say you’re about to decide your next vacation plans, which you’re reasonably confident will be satisfying. Should you go for one week or two?

Making the question more interesting, Kahneman asks, would you pick the same vacations if you knew that after, all the pictures would be destroyed and you’d take an amnesiac drug forcing you to forget ever having taken it?

When I talk about the pursuit of the ideal life here on this blog, this revelation asks the question, what constitutes the ideal life? Is it our moment-to-moment experiences or simply the narrative we weave afterward?

The Tyranny of the Biographer
The difficulty with designing a life, is that your experiencing self doesn’t get a vote. It’s only the biographer, the part of yourself that remembers the past and predicts the future that gets a say in what careers you pick, vacations you choose and people you spend time with.

This doesn’t really seem fair. What you actually would write about your life after it has been lived is merely paper and some ink. It’s the slivers of time that pass through our consciousness that feel important.

This problem goes beyond the common experience of doing something for the purpose of talking about it later. Such as people who run marathons to “say they did it.” The reason our experiences don’t get a vote, is because they’ve already been taken over by the inner biographer.

We don’t base future decisions on experiential happiness because we don’t have access to anything but this sliver of now. Everything else has been converted to memories, and subject to all of the biases of the storyteller.

Experiencing the Ideal Life, Instead of Simply Narrating It
I don’t know about you, but I find this biographical tyranny unacceptable. I wouldn’t want to invest a lot of time designing a life that I can tell myself is great, but is lousy when I actually experience it.


As with all cognitive biases, I don’t think there is an easy solution. To err is human, and unfortunately, so is to err systematically.

However, I have tried to add a few broad ways of thinking about my life to avoid the most obvious traps. Here are a few of the mental habits I’m trying to foster to escape biographical tyranny:

#1 – Stop and Observe the Now
Kahneman explains that the middle moments are often washed out in memory. We accentuate when things start, when they end, or when something dramatic changes.

One counteracting force is simply to ask yourself how you feel at the current moment. Not a whole-life assessment, but a stopwatch checkup on your instantaneous mental state. Doing this, I believe, has helped me better recall how I’ve felt during a period, instead of just the end.


Eckhart Tolle has sold thousands of books preaching the pseudo-spiritual wisdom to stay in the present moment. Ignoring the fact it is mostly a rehash of millennia-old advice in a new-agey package, I feel some popularity of this comes back to the issue of biographical tyranny. We are so frequently absorbed in the thoughts of our life in totality, future worries, past regrets, that we fail to pay attention to the slivers of now that actually constitute our lives.

#2 – Emphasize Rewarding Routines, over Brief Events
Since middle-moments and sameness are washed out in the biography of our lives, it makes sense to deliberately weight these higher in our decisions for the future.

To put it another way, it makes more sense to focus on how your lifestyle affects your routine, than one-time events. For example, in thinking of my stay here in France, I’m likely to remember my amazing weekend in Barcelona, or the brief relationships I’ve had.

Those experiences, however positive, composed a lot less time than, say, being in class or walking to get groceries. The things we do every day, if they contribute positively to our well-being or detract from it, may be thousands of times more important than shorter events.

When planning for the future, this means I should spend a lot more time on decisions that enable me to avoid the hundreds of hours of boring class time, instead of my brief, but frustrating experience without electricity, for example. One may be more memorable, but the other occupies far more of my experiential life.

#3 – Create a Way of Living, Instead of a Goal
“Life is a journey, not a destination” is a clichÃ©, but it’s still true. That’s one of the reasons I’m a fan of the lifestyle design concept. Because it turns around the typical accomplishment-oriented ambitions many people have towards one that focuses on how you actually live all the moments in-between.

Now, if I pick new challenges, I make sure to pick ones that will be enjoyable on the way to my destination. Ideally, I also try to pick goals that, if reached, will improve the way I’m able to live.

Setting up an online business to pay for all my expenses had been a major one for me, since it allows me the freedom to only work on things I’m interested in. Getting in decent physical shape and eating healthy was another one, as it has given me more energy.
"""
result=chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()