# Noita Technical Documentation

## Overview

Noita is a permanent roguelike game where every pixel in the world has physical properties. The game features a custom physics engine called "Falling Everything" that simulates complex cellular automata behaviors.

### Game Description
- **Engine**: Custom physics engine called "Falling Everything"
- **Core Concept**: Every pixel has physical properties (solid, liquid, gas, fire, etc.)
- **Gameplay**: Players explore procedurally generated worlds, collect materials, cast spells with wands, and survive against various hazards
- **Development**: Built with focus on emergent behaviors and sandbox-style gameplay

## Technical Architecture

### Engine Foundation
The game is based on a falling sand style simulation. Essentially, it's complex cellular automata. Every pixel in the game follows rather simple rules, but when you combine them together you get surprising and unexpected results.

For example, liquid pixels first check if they can go down. If not, they check left and right to see if they can move that way. With those rules, you get a rudimentary 2D liquid. To make liquids have different densities you just compare the densities when figuring out if you can go down (and then swap the pixels).

The fire is kind of simple as well. When something is on fire it will look in a random direction to see if it can ignite that pixel. Oil is flammable, but water is not so the fire will spread in the oil, but stop when it hits the water. If the fire tries to ignite the water it will convert into steam instead.

For gases (like steam), you can think of the rules as for the inverted water. First, you check if you can go up – if so, move that way. If not, check if you can move left or right.

The wood that burns is easy, you don't really do anything. If set on fire, you wait until the fire has been burning long enough and then you destroy that pixel.

The rules are in practice a bit more complicated, but that's the basic logic.

## Core Systems

### Material System Mechanics
The game features a comprehensive material system where every material has specific behaviors:

**Basic Material Properties:**
- Sand, Liquids, Gases, Fire, Wood, Water, Oil, and more
- Each material follows simple physics rules
- Complex emergent behaviors from simple rules

**Material Interactions:**
- Fire spreads to flammable materials (like oil) but stops at non-flammable (like water)
- Water can freeze to ice and melt back to water
- Materials can combine to create new substances
- Different densities affect how materials fall

### Explosions & Collapses
We have several materials that stick to the ceiling and fall infrequently, but most of the materials that are static like stone don't have those rules applied. So what we do is that when there's an explosion large enough we look at the pixels that the explosion collided with. If they happen to be "good candidates", we will turn those into collapsing sand materials.

Then, we also have rigid bodies in the game. Each pixel in a rigid body knows that it belongs to that rigid body and location inside that rigid body. So when a rigid body is updated, the pixel gets its position that way. If a pixel that belongs to a rigid body gets destroyed, the simulation recalculates the shape for the rigid body (or rigid bodies if the shape is now cut in two or more pieces).

To make chunks fall down, we do what we did with the falling pixels, but in this case, we do it with rigid bodies. We look for a good place to collapse things and then try to figure if we can carve out shapes. If we can then we create rigid bodies out of those materials and let them fall.

### Procedural Generation of the World
The world is procedurally generated, but we do have an overall structure that stays the same, mostly to give the game good pacing and for us to control how the game plays. The basic structure of the world is the following: there is the first area (mines) and underneath that, there's the second area, and so forth.

We can't say what's at the very bottom as it's a bit spoilery, but technically, the engine does allow you to go very very very deep. We had a few infinite fall moments during the development.

We tried a few different ways of doing procedural generation. At the moment, we use a few different techniques, but the main part of it uses Sean Barrett's Herringbone Wang Tiles. Essentially, we have a series of pre-made pieces that are like bricks. They're laid together in a Herringbone pattern. The Herringbone pattern disguises the edge of the pieces better than the usual square pieces in which you can easily see the "seam". Within those pieces, we further randomize things, like with a 50% of putting a random enemy here, a random item there and so forth. We've been pretty happy with how well the technique works for a game because it allows us to have quite a bit of control on how things are laid out and still generates a lot of new and varied situations.

### Wand System
The wand system allows the players to modify and create their own spells. The way it works is that you as the player find the randomly generated wands all around the world. Then, in between each area, you get to modify those wands.

The system was inspired by the deck building games like Donald X. Vaccarino's Dominion. When the player modifies the wands they get to move the spells around between the wands, much like the cards in a deck building game. When the player shoots their wand it draws up the "cards" from the wand and applies the effects. It sounds easy in theory, but in practice, it creates a lot of interesting situations and combinations.

### Natural Phenomena
There are a few slow reaction things in our game. We have grass that slowly grows on top of the soil. There's also moss that slowly grows on the rocks. The water freezes. Snow melts. The corpses decompose.

They're all really cool, but when you're playing the game you rarely notice them. So we haven't really spent too much time on adding stuff like that into the game.

### Optimization of the Falling Sand Simulation
We've kept performance in mind during all the development. We use profiling tools to know how much time the stuff is taking and catch early on if something is taking too many precious CPU cycles.

There are a few techniques we use to keep the updating of the world at a reasonable pace. This goes a bit deeper into the falling sand simulation. In order to make the falling sand simulation work, you have to update the world from the bottom up. The reason is that if you don't do that only the pixels at the very bottom will fall down. The next frame, the pixels above those will move down and so forth. But if you update it from the bottom up, the sequence of the falling rows will be consistent.

Now, having to update the whole world in order to make the falling sand simulation is a really bad fit for multi-threading. The way that's been solved in Noita is that the world is divided into tinier 64×64 chunks. Each chunk keeps a dirty rect to indicate which pixels need to be updated. So when something is moving in a chunk it marks itself to be updated. This divides the dirty rect into a bunch of tinier dirty rects and those pixels are iterated over. Which is already a huge benefit because it cuts down on the number of pixels to update. But if you were to just naively multi-thread those chunks, some problems would occur when there's a pixel that leaves its 64×64 area and wanders into the range of another chunk. You don't want to update the pixel multiple times, so you can just add a counter that says "I've been updated this frame, so don't update me if the frame is the same". But then you'd have to add atomics or locks to make sure that the value is up to date on all threads. And that would slow things down a lot.

The way we've solved this problem is to update the world in a checker pattern of sorts. We do this 4 times. In the first round, we pick every other 64×64 chunk, and the pixels inside the chunks are allowed to move within that 64×64 area plus 32 pixels in each cardinal direction. That creates little crosses. This ensures that every pixel inside that area is only updated by that thread. Then we do this 4 times. That way every thread can safely update everything inside it, and we don't worry that other threads are accessing their pixels.

## Development Process

### Audio Department Introduction
[Music] we're going to be talking about exploring the tech and design of Noita because the game is not out yet. We had some trouble getting the audio working so it's not again this is the curse of mine [Music] [Music] so uh thank you. This is essentially us uh there's three of us and then this is our uh audio department here and uh so the talk is going to be divided into two parts and the first part we're going to talk about the tech stuff and then it's going to be the design stuff. And I'm just going to say right now that the design part is there's not probably going to be enough time to go into too much detail about it so it's going to be a bit more rambly but for the tech so we have a custom engine that's called falling everything and you can kind of see from the trailer we can do sand and we can do liquids of all kinds and then we can do gases as well and and for the last thing is like we also have like rigid bodies in there and essentially I'm going to talk to how we've sort of done all of that and how that is sort of being accomplished and we're going to start at the basics by traveling back into a much better time period into the 90s when you could type into a text editor and characters would appear immediately. So back back in the 90s I got started in quick basic and I discovered in quick basic that you can do this thing where you can put a pixel under the screen by pset and you can ask for a color of a pixel by point by so you point at something and you ask what the color of that is and with this amazing technology I was able to essentially write a sand simulation. What's happening here is it's testing that if there's a sand pixel here it's looking at the line below it if that's empty then move down and then if that's occupied it looks to the left and the right to the down left and the right and it moves in that direction and with that you essentially ge...

### Games Now Lecture Introduction
Welcome to the lecture. My name is Onika School Dima, and this is Games Now. As you know, Games Now is an open lecture series which means that you can always invite your friends here and we also stream it so anybody can access it. This is about game development and we run it about once a month at a university in order to take the full advantage of what we can offer. Please follow our social media channels meaning YouTube where we are currently streaming and Facebook where we're also streaming but a bit less. Well today we have one of the best finish in the houses presenting their work. So NOLA games is paid reporter only her Ella and are we take are today talking about Noida and specifically all the fun that has been put into making it. I hope it's an exciting times are is living it yep life so he's an airplane maybe already landed so we're not absolutely sure if we can get him today here for the presentations but we'll pull him in as soon as he arrives. Without further ado I'll give the stage to the first speaker of the day. That report, thank you. Yes, okay. I think I'll just continue playing this. How long did I have to kill time for 15 minutes before our bee arrived. So actually how many of you have played noise or know about it? Okay, so fairly fairly big portion. So hopefully this early demo here kind of showed a little bit about the games. I don't have to go into detail about like what you do in this game and so forth. I think that's okay. Or is there someone who wants to know more about the game but like know more about the games specifically as in the game how the game plays? You really want to know more gold? Do you want the Lord explanation or what you know that the lore is Lloris forbidden. I'm not allowed to talk about it secrets. Okay, so I'll briefly tell a little bit about what Norte is that and how it plays. So it is a permanent roguelike in this world where every pixel has physical properties. So for example here you can you can see here you're my inventory it tells you what ma...