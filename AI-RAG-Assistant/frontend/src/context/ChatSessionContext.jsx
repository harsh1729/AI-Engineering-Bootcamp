import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  fetchChatMessages,
  fetchChats,
  mapPersistedMessages,
} from "../api/chatHistoryApi";
import { useAuth } from "./AuthContext";

const ChatSessionContext = createContext(null);

export function ChatSessionProvider({ children }) {
  const { user, isLoading: isAuthLoading } = useAuth();
  const [chatProvider, setChatProvider] = useState(null);
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const sessionGenerationRef = useRef(0);

  const bumpSessionGeneration = useCallback(() => {
    sessionGenerationRef.current += 1;
    return sessionGenerationRef.current;
  }, []);

  const refreshChats = useCallback(
    async (
      providerValue,
      generation = sessionGenerationRef.current,
      useAuth = Boolean(user),
    ) => {
      if (!providerValue) {
        setChats([]);
        return [];
      }

      const data = await fetchChats(providerValue, { useAuth });
      if (generation !== sessionGenerationRef.current) {
        return [];
      }

      const nextChats = data.chats ?? [];
      setChats(nextChats);
      return nextChats;
    },
    [user],
  );

  const loadChatMessages = useCallback(
    async (
      chatId,
      providerValue,
      generation = sessionGenerationRef.current,
      useAuth = Boolean(user),
    ) => {
      const data = await fetchChatMessages(chatId, providerValue, { useAuth });
      if (generation !== sessionGenerationRef.current) {
        return;
      }

      setMessages(mapPersistedMessages(data.messages ?? []));
    },
    [user],
  );

  const openLatestChatOrBlank = useCallback(
    async (
      providerValue,
      generation = sessionGenerationRef.current,
      useAuth = Boolean(user),
    ) => {
      if (!providerValue) {
        return;
      }

      setIsLoadingSessions(true);
      setIsLoadingMessages(true);

      try {
        const loadedChats = await refreshChats(providerValue, generation, useAuth);
        if (generation !== sessionGenerationRef.current) {
          return;
        }

        if (loadedChats.length === 0) {
          setActiveChatId(null);
          setMessages([]);
          return;
        }

        const latestChatId = loadedChats[0].id;
        setActiveChatId(latestChatId);
        await loadChatMessages(latestChatId, providerValue, generation, useAuth);
      } catch {
        if (generation !== sessionGenerationRef.current) {
          return;
        }

        setChats([]);
        setActiveChatId(null);
        setMessages([]);
      } finally {
        if (generation === sessionGenerationRef.current) {
          setIsLoadingMessages(false);
          setIsLoadingSessions(false);
        }
      }
    },
    [loadChatMessages, refreshChats, user],
  );

  const selectChat = useCallback(
    async (chatId, providerValue) => {
      if (!providerValue || chatId === activeChatId || isSending || isLoadingMessages) {
        return;
      }

      setActiveChatId(chatId);
      setIsLoadingMessages(true);

      try {
        await loadChatMessages(chatId, providerValue, sessionGenerationRef.current, Boolean(user));
      } catch (error) {
        setMessages([]);
        window.alert(error.message || "Could not load this chat.");
      } finally {
        setIsLoadingMessages(false);
      }
    },
    [activeChatId, isLoadingMessages, isSending, loadChatMessages, user],
  );

  const startNewChat = useCallback(() => {
    if (isSending || isLoadingMessages) {
      return;
    }

    setActiveChatId(null);
    setMessages([]);
  }, [isLoadingMessages, isSending]);

  useEffect(() => {
    if (isAuthLoading) {
      return;
    }

    const generation = bumpSessionGeneration();
    const useAuth = Boolean(user);

    setChats([]);
    setActiveChatId(null);
    setMessages([]);

    if (!chatProvider) {
      return;
    }

    openLatestChatOrBlank(chatProvider, generation, useAuth);
  }, [user, isAuthLoading, chatProvider, openLatestChatOrBlank, bumpSessionGeneration]);

  const activeChatTitle = useMemo(() => {
    if (activeChatId) {
      const activeChat = chats.find((chat) => chat.id === activeChatId);
      return activeChat?.title ?? null;
    }

    if (messages.length > 0) {
      return "Draft conversation";
    }

    return "New conversation";
  }, [activeChatId, chats, messages.length]);

  const sessionBusy = isSending || isLoadingSessions || isLoadingMessages;

  const value = useMemo(
    () => ({
      chatProvider,
      setChatProvider,
      chats,
      activeChatId,
      activeChatTitle,
      messages,
      setMessages,
      isLoadingSessions,
      isLoadingMessages,
      isSending,
      setIsSending,
      sessionBusy,
      refreshChats,
      openLatestChatOrBlank,
      selectChat,
      startNewChat,
      setActiveChatId,
    }),
    [
      chatProvider,
      chats,
      activeChatId,
      activeChatTitle,
      messages,
      isLoadingSessions,
      isLoadingMessages,
      isSending,
      sessionBusy,
      refreshChats,
      openLatestChatOrBlank,
      selectChat,
      startNewChat,
      setActiveChatId,
    ],
  );

  return (
    <ChatSessionContext.Provider value={value}>{children}</ChatSessionContext.Provider>
  );
}

export function useChatSession() {
  const context = useContext(ChatSessionContext);
  if (!context) {
    throw new Error("useChatSession must be used within ChatSessionProvider");
  }
  return context;
}
