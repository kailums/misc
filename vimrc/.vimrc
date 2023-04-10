" Set encoding
set encoding=utf-8

" Enable syntax highlighting
syntax enable

" Set number of spaces for indentation
set smarttab
set tabstop=4
set shiftwidth=2
set expandtab

set ai
set si

" Show line numbers
set number

set hlsearch

" Enable mouse support
" set mouse=a

" Set color scheme
" colorscheme desert

" Map jj to escape in insert mode


" Remember cursor position
if has("autocmd")
  autocmd BufReadPost *
    \ if line("'\"") > 0 && line("'\"") <= line("$") |
    \   exe "normal! g`\"" |
    \ endif
endif
