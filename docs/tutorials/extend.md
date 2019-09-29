# Extend Detectron2's Defaults

__Research is all about doing things in new ways__.
This brings a tension in how to create abstraction in code,
which is a big challenge for any research engineering project of a significant size:

1. On one hand, it needs to have very thin abstraction to allow the possibility of doing
   everything in new ways. It should be reasonably easy to break existing
   abstraction and replace it with new ones.

2. On the other hand, such a project also needs a reasonably high-level
   abstraction, so that users can easily do things in standard ways,
   without worrying too much about the details which only a few researchers care about.
   
In detectron2, there are two types of interfaces that address this tension together:

1. Functions and classes that take only a "config" argument (or plus minimal
   number of extra arguments).

   Such functions and classes implement
   the "standard default" behavior: it will take what it needs from the giant
   config and do the "standard" thing. 
   Users only need to load a standard config and pass it around, without having to worry about
   which twenty arguments are used and what they all mean.
   
2. Functions and classes that has well-defined explicit arguments. 

   Each of them is a small building block of the entire system.
   They require users' effort to stitch together, but can be stitched together in more flexible ways.
   When you need to implement something different from the "standard default"
   ones included in detectron2, these well-defined components can be reused.
   

If you only need the standard behavior, the [Beginner's Tutorial](beginner)
should suffice. If you need to extend detectron2 to your own needs,
see the following tutorials for more details:

* Detectron2 includes a few standard datasets, but you can [use custom ones](datasets).
* Detectron2 contains the standard logic that [creates a data loader from a
  dataset](data_loading), but you can write your own versions as well.
* Detectron2 implements many standard detection models, and provide ways for you
  to overwrite its behaviors. (See [models](models))
* Detectron2 provides a default training loop that is good for common training tasks.
  You can customize it with hooks, or write your own loop instead. (See [training](training))
